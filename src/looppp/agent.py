"""The agent loop (runs on WSL).

Per generation: build context -> ask the model for N candidates -> precheck
locally -> submit to the queue -> wait for the remote worker -> pick the best
graded candidate as the next parent -> archive improvements -> stop checks.

State lives in <state_dir>/<loop_id>/ so a loop can be resumed after a crash
or a long worker outage (``looppp run --loop-id <id>``).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from looppp import contract as c
from looppp.archive import Archive
from looppp.config import Config
from looppp.contract import CandidateSpec, GradeResult
from looppp.llm import LLM, EmptyCompletionError
from looppp.parsing import ResponseFormatError, parse_llm_response
from looppp.precheck import precheck
from looppp.problems import Problem
from looppp.prompts import (MODE_EXPLOIT, MODE_EXPLORE, MODE_SEED, HistoryItem, build_propose_messages,
                             build_repair_messages)
from looppp.queue.base import Queue
from looppp.tracker import Tracker

# Attempt statuses that never reach the queue.
LOCAL_FAILED = "local_failed"
OPEN_STATUSES = (c.STATUS_PENDING, c.STATUS_RUNNING, c.STATUS_SUBMITTING)
MAX_CONSECUTIVE_LLM_FAILURES = 3


@dataclass
class Attempt:
    generation: int
    index: int
    hypothesis: str
    status: str
    model: str
    code_sha256: str = ""
    code_file: str = ""
    candidate_id: str | None = None
    parent: str | None = None
    result: dict | None = None
    alive_wait_seconds: float = 0.0
    created_ts: float = field(default_factory=time.time)
    mode: str = MODE_EXPLOIT             # exploit | explore | seed
    llm_seconds: float = 0.0             # model time for this candidate (all repair rounds)
    completion_tokens: int = 0
    started_ts: float | None = None      # before the first model call
    finished_ts: float | None = None     # when the grade came back

    @property
    def grade(self) -> GradeResult | None:
        return GradeResult.from_summary(self.result) if self.result else None

    @property
    def peak(self) -> float | None:
        g = self.grade
        return g.peak_fraction if g and g.scored else None

    @property
    def label(self) -> str:
        return f"gen {self.generation}.{self.index}"


@dataclass
class LoopOutcome:
    loop_id: str
    stop_reason: str
    generations: int
    attempts: int
    best: Attempt | None


class WorkerDown(RuntimeError):
    pass


class AgentLoop:
    def __init__(self, cfg: Config, problem: Problem, queue: Queue, llm: LLM, tracker: Tracker,
                 archive: Archive, loop_id: str, deck_commit: str, *,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 log: Callable[[str], None] = print, seed: tuple[str, str] | None = None):
        self.cfg, self.ls, self.qs = cfg, cfg.loop, cfg.queue
        self.seed = seed  # (code, label): a known kernel graded first and used as the starting parent
        self.problem, self.queue, self.llm, self.tracker, self.archive = problem, queue, llm, tracker, archive
        self.loop_id, self.deck_commit = loop_id, deck_commit
        self.clock, self.sleep, self.log = clock, sleep, log
        self.dir = cfg.path(self.ls.state_dir) / loop_id
        (self.dir / "code").mkdir(parents=True, exist_ok=True)
        self.attempts: list[Attempt] = []
        self.best_gen = -1
        self._down_since: float | None = None
        self._stop_requested = False
        self._llm_failures = 0
        self.activity, self.activity_since = "starting", time.time()  # read by the notebook dashboard

    def _set_activity(self, text: str) -> None:
        self.activity, self.activity_since = text, time.time()

    def request_stop(self) -> None:
        """Ask the loop to stop at the next safe point (between model calls or polls). Candidates
        already submitted stay in the queue; a later `run()` with the same loop id waits for them."""
        self._stop_requested = True

    # --- persistence ----------------------------------------------------------
    def _state_file(self) -> Path:
        return self.dir / "attempts.json"

    def _save(self) -> None:
        tmp = self._state_file().with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(a) for a in self.attempts], indent=1, default=str))
        os.replace(tmp, self._state_file())

    def _load(self) -> None:
        if self._state_file().exists():
            self.attempts = [Attempt(**a) for a in json.loads(self._state_file().read_text())]
        meta = self.dir / "meta.json"
        if not meta.exists():
            meta.write_text(json.dumps({"loop_id": self.loop_id, "problem": self.problem.name,
                                        "model": self.llm.model, "deck_commit": self.deck_commit,
                                        "loop": asdict(self.ls)}, indent=2))

    def _code(self, a: Attempt) -> str:
        return (self.dir / a.code_file).read_text()

    # --- selection --------------------------------------------------------------
    def _best_attempt(self) -> Attempt | None:
        scored = [a for a in self.attempts if a.peak is not None]
        return max(scored, key=lambda a: a.peak) if scored else None

    def _best_peak(self) -> float | None:
        own = self._best_attempt()
        archived = self._archived()
        vals = [v for v in (own.peak if own else None, archived[1]["peak_fraction"] if archived else None)
                if v is not None]
        return max(vals) if vals else None

    def _archived(self) -> tuple[str, dict] | None:
        cur = self.archive.best(self.problem.name)
        if cur and cur[1].get("deck_commit") == self.deck_commit:
            return cur
        return None

    def _top_parents(self, k: int) -> list[tuple[str, str, GradeResult | None, str | None]]:
        """Up to k distinct graded kernels, best first (the archived best counts if it beats them)."""
        seen, own = set(), []
        for a in sorted((a for a in self.attempts if a.peak is not None), key=lambda a: a.peak, reverse=True):
            if a.code_sha256 not in seen:
                seen.add(a.code_sha256)
                own.append(a)
        parents = [(self._code(a), f"{a.label} ({a.hypothesis})", a.grade, a.candidate_id) for a in own[:k]]
        archived = self._archived()
        if archived and (not own or archived[1]["peak_fraction"] > own[0].peak):
            parents = [self._parent()] + parents[:k - 1]
        return parents or [self._parent()]

    def _stagnant_generations(self, generation: int) -> int:
        """Generations before *generation* since the best score last rose by >= min_rel_improvement."""
        last_significant, best = -1, None
        for g in sorted({a.generation for a in self.attempts if a.generation < generation}):
            gen_best = max((a.peak for a in self.attempts if a.generation == g and a.peak is not None), default=None)
            if gen_best is None:
                continue
            if best is None or gen_best >= best * (1 + self.ls.min_rel_improvement):
                last_significant = g
            best = gen_best if best is None else max(best, gen_best)
        return max(0, (generation - 1) - last_significant)

    def _tried_approaches(self, limit: int = 25) -> list[str]:
        out = []
        for a in self.attempts:
            if a.mode == MODE_SEED or not a.code_file:
                continue
            g = a.grade
            outcome = (f"{a.peak:.4f}" if a.peak is not None else (g.fail_stage if g and g.fail_stage else a.status))
            out.append(f"{a.label} [{outcome}]: {a.hypothesis[:140]}")
        return out[-limit:]

    def _parent(self) -> tuple[str, str, GradeResult | None, str | None]:
        own, archived = self._best_attempt(), self._archived()
        if own and (not archived or own.peak >= archived[1]["peak_fraction"]):
            return self._code(own), f"{own.label} ({own.hypothesis})", own.grade, own.candidate_id
        if archived:
            code, meta = archived
            res = GradeResult(correct=True, peak_fraction=meta["peak_fraction"],
                              shape_fractions=meta.get("shape_fractions", []))
            return code, f"archived best from {meta.get('loop_id')} {meta.get('generation')}.{meta.get('index')}", \
                res, meta.get("candidate_id")
        return self.problem.reference, "reference.py (naive baseline; not itself a valid kernel)", None, None

    # --- worker liveness ---------------------------------------------------------
    def _worker_alive(self) -> bool:
        try:
            age = self.queue.worker_heartbeat_age()
        except Exception as e:  # noqa: BLE001 - treat an unreachable queue like a dead worker
            self.log(f"heartbeat query failed: {type(e).__name__}: {e}")
            age = None
        alive = age is not None and age < self.qs.heartbeat_stale_seconds
        now = self.clock()
        if alive:
            if self._down_since is not None:
                self.tracker.alert("looppp: worker back", f"{self.loop_id}: worker heartbeat resumed; continuing.")
            self._down_since = None
        else:
            if self._down_since is None:
                self._down_since = now
                seen = "never reported" if age is None else f"last heartbeat {age / 60:.1f} min ago"
                self.tracker.alert("looppp: worker offline",
                                   f"{self.loop_id}: no fresh worker heartbeat ({seen}). The loop is paused and resumes "
                                   "automatically (distributed mode: restart the molab evaluator notebook).")
            if now - self._down_since > self.qs.max_worker_down_hours * 3600:
                raise WorkerDown(f"worker offline for more than {self.qs.max_worker_down_hours} h")
        return alive

    def _await_worker(self) -> None:
        if not self._worker_alive():
            self._set_activity("waiting for a live grader heartbeat")
        while not self._stop_requested and not self._worker_alive():
            self.sleep(self.qs.poll_seconds)

    # --- proposing ------------------------------------------------------------------
    def _history(self) -> list[HistoryItem]:
        done = [a for a in self.attempts if a.status not in OPEN_STATUSES]
        return [HistoryItem(a.generation, a.index, a.hypothesis, a.status, a.grade)
                for a in done[-self.ls.history_window:]]

    def _failure_logs(self, generation: int) -> list[tuple[HistoryItem, str]]:
        out = []
        for a in self.attempts:
            g = a.grade
            if a.generation != generation or g is None or g.scored or a.status == LOCAL_FAILED:
                continue
            log = (g.bench_tail if g.correct else g.check_tail) or g.fail_reason
            out.append((HistoryItem(a.generation, a.index, a.hypothesis, a.status, g), log[-3000:]))
        return out[:2]

    def _store_code(self, generation: int, index: int, code: str) -> str:
        rel = f"code/g{generation:03d}-i{index}.py" if generation >= 0 else f"code/seed-i{index}.py"
        (self.dir / rel).write_text(code)
        return rel

    def _propose_one(self, generation: int, index: int, n: int, parent, siblings: list[str],
                     mode: str = MODE_EXPLOIT, tried: list[str] | None = None, stagnant: int = 0) -> Attempt:
        parent_code, parent_label, parent_result, parent_id = parent
        messages = build_propose_messages(
            self.problem, parent_code=parent_code, parent_label=parent_label, parent_result=parent_result,
            history=self._history(), failures=self._failure_logs(generation - 1),
            sibling_hypotheses=siblings, index=index, n=n, best_peak=self._best_peak(), mode=mode,
            tried=tried, stagnant_generations=stagnant, min_rel_improvement=self.ls.min_rel_improvement,
        )
        started, llm_seconds, tokens = time.time(), 0.0, 0
        timing = lambda a: self._stamp(a, mode, started, llm_seconds, tokens)  # noqa: E731
        hypothesis, code, errors = "(no reply)", "", ["no reply"]
        for repair in range(self.ls.precheck_repairs + 1):
            what = "waiting for model" if repair == 0 else f"waiting for model (repair {repair})"
            self._set_activity(f"gen {generation}.{index} (candidate {index + 1}/{n}): {what}")
            self.log(f"gen {generation}.{index}: {what} ({self.llm.model}, prompt ~{sum(len(m['content']) for m in messages) // 4} tokens)")
            t_call = time.monotonic()
            try:
                reply = self.llm.complete(messages)
                self._llm_failures = 0
            except Exception as e:  # noqa: BLE001 - one bad call costs one candidate, not the session
                llm_seconds += time.monotonic() - t_call
                self._llm_failures += 1
                reason = str(e) if isinstance(e, EmptyCompletionError) else f"{type(e).__name__}: {e}"
                if self._llm_failures >= MAX_CONSECUTIVE_LLM_FAILURES:
                    raise RuntimeError(f"{self._llm_failures} model calls failed in a row; last: {reason}") from e
                return timing(self._local_failure(generation, index, "(model call failed)", c.STAGE_LLM, reason, ""))
            llm_seconds += time.monotonic() - t_call
            tokens += reply.completion_tokens or 0
            self.tracker.log({"llm/prompt_tokens": reply.prompt_tokens, "llm/completion_tokens": reply.completion_tokens,
                              "llm/attempts": reply.attempts, "generation": generation})
            try:
                hypothesis, code = parse_llm_response(reply.text)
                errors = precheck(code, self.problem.forbidden).errors
            except ResponseFormatError as e:
                errors = [str(e)]
            if not errors:
                break
            if repair < self.ls.precheck_repairs:
                messages = build_repair_messages(messages, reply.text, errors)

        if errors:
            return timing(self._local_failure(generation, index, hypothesis, c.STAGE_PRECHECK, "; ".join(errors), code))

        sha = c.sha256_text(code)
        dup = next((a for a in self.attempts if a.code_sha256 == sha and a.candidate_id), None)
        if dup:
            return timing(self._local_failure(generation, index, hypothesis, c.STAGE_DUPLICATE,
                                              f"identical to {dup.label}", code))
        return timing(self._submit_attempt(generation, index, hypothesis, code, parent_id, mode, self.llm.model))

    @staticmethod
    def _stamp(a: Attempt, mode: str, started: float, llm_seconds: float, tokens: int) -> Attempt:
        a.mode, a.started_ts, a.llm_seconds, a.completion_tokens = mode, started, round(llm_seconds, 1), tokens
        return a

    def _submit_attempt(self, generation: int, index: int, hypothesis: str, code: str, parent_id: str | None,
                        mode: str, model: str) -> Attempt:
        sha = c.sha256_text(code)
        attempt = Attempt(generation, index, hypothesis, c.STATUS_SUBMITTING, model, sha,
                          self._store_code(generation, index, code), parent=parent_id, mode=mode)
        spec = CandidateSpec(problem=self.problem.name, deck_commit=self.deck_commit, loop_id=self.loop_id,
                             generation=generation, index=index, model=model, hypothesis=hypothesis,
                             code_sha256=sha, parent=parent_id, mode=mode)
        for try_no in range(3):
            try:
                attempt.candidate_id = self.queue.submit(spec, code)
                attempt.status = c.STATUS_PENDING
                break
            except Exception as e:  # noqa: BLE001
                self.log(f"submit failed ({try_no + 1}/3): {type(e).__name__}: {e}")
                self.sleep(min(60.0, self.qs.poll_seconds))
        else:
            attempt.status = LOCAL_FAILED
            attempt.result = GradeResult(correct=False, fail_stage=c.STAGE_SUBMIT,
                                         fail_reason="queue submit failed 3 times").to_summary()
        self.log(f"{attempt.label}: {attempt.status} - {hypothesis}")
        return attempt

    def _local_failure(self, generation, index, hypothesis, stage, reason, code) -> Attempt:
        self.log(f"gen {generation}.{index}: {stage} - {reason[:200]}")
        return Attempt(generation, index, hypothesis, LOCAL_FAILED, self.llm.model,
                       c.sha256_text(code) if code else "", self._store_code(generation, index, code) if code else "",
                       result=GradeResult(correct=False, fail_stage=stage, fail_reason=reason).to_summary())

    # --- waiting --------------------------------------------------------------------
    def _wait(self) -> None:
        self._set_activity("waiting for grading")
        last = self.clock()
        while not self._stop_requested:
            open_ = [a for a in self.attempts if a.status in OPEN_STATUSES and a.candidate_id]
            if not open_:
                return
            alive = self._worker_alive()
            now = self.clock()
            dt, last = now - last, now
            for a in open_:
                try:
                    rec = self.queue.get(a.candidate_id)
                except Exception as e:  # noqa: BLE001
                    self.log(f"status query failed for {a.label}: {type(e).__name__}: {e}")
                    continue
                a.status = rec.status
                if rec.terminal:
                    a.result = rec.result.to_summary() if rec.result else a.result
                    a.finished_ts = time.time()
                    g = a.grade
                    self.log(f"{a.label}: {rec.status} correct={g.correct if g else None} "
                             f"peak={g.peak_fraction if g else None} stage={g.fail_stage if g else None}")
                elif alive:
                    a.alive_wait_seconds += dt
                    if a.alive_wait_seconds > self.qs.wait_timeout_hours * 3600:
                        self.queue.abandon(a.candidate_id, "agent wait timeout")
                        a.status = c.STATUS_ABANDONED
                        a.result = GradeResult(correct=False, fail_stage=c.STAGE_TIMEOUT,
                                               fail_reason="not graded within wait_timeout_hours").to_summary()
            self._save()
            if any(a.status in OPEN_STATUSES for a in open_):
                self.sleep(self.qs.poll_seconds)

    # --- main -------------------------------------------------------------------------
    def _stop_reason(self, generation: int, started: float) -> str | None:
        if self._stop_requested:
            return "stopped"
        best = self._best_peak()
        if self.ls.target_peak_fraction is not None and best is not None and best >= self.ls.target_peak_fraction:
            return "target_reached"
        if generation >= self.ls.max_generations:
            return "max_generations"
        if self.clock() - started > self.ls.wall_budget_hours * 3600:
            return "wall_budget"
        if generation > 0 and (generation - 1) - self.best_gen >= self.ls.patience:
            return "patience"
        return None

    def _finish_generation(self, generation: int, previous_best: float | None) -> None:
        gen_attempts = [a for a in self.attempts if a.generation == generation]
        gen_scored = [a for a in gen_attempts if a.peak is not None]
        best = self._best_attempt()
        improved = best is not None and best.generation == generation and \
            (previous_best is None or best.peak > previous_best)
        if improved and best.mode == MODE_SEED:
            self.best_gen = generation
            self.log(f"seed graded: {best.peak:.4f} (not archived: it is not this loop's work)")
        elif improved:
            self.best_gen = generation
            meta = {"peak_fraction": best.peak, "shape_fractions": best.grade.shape_fractions,
                    "candidate_id": best.candidate_id, "model": best.model, "hypothesis": best.hypothesis,
                    "loop_id": self.loop_id, "generation": best.generation, "index": best.index,
                    "deck_commit": self.deck_commit}
            if self.archive.maybe_update(self.problem.name, self._code(best), meta):
                self.log(f"archive updated: {self.problem.name} -> {best.peak:.4f}")
        stages: dict[str, int] = {}
        for a in gen_attempts:
            g = a.grade
            if g and not g.scored and g.fail_stage:
                stages[g.fail_stage] = stages.get(g.fail_stage, 0) + 1
        self.tracker.log({
            "generation": generation,
            "gen_explore": sum(1 for a in gen_attempts if a.mode == MODE_EXPLORE),
            "gen_best_peak_fraction": max((a.peak for a in gen_scored), default=None),
            "best_peak_fraction": self._best_peak(),
            "gen_scored": len(gen_scored), "gen_attempts": len(gen_attempts),
            **{f"gen_fail/{k}": v for k, v in stages.items()},
        })

    def run(self) -> LoopOutcome:
        self._load()
        started = self.clock()
        best = self._best_attempt()
        self.best_gen = best.generation if best else -1
        generation = max((a.generation for a in self.attempts), default=-1) + 1
        stop_reason = "unknown"
        try:
            resumed_open = [a for a in self.attempts if a.status in OPEN_STATUSES and a.candidate_id]
            if resumed_open:  # resume: finish the generation that was in flight before a restart
                self.log(f"resuming: waiting for {len(resumed_open)} candidates from a previous session")
                self._wait()
                self._finish_generation(generation - 1, previous_best=None)
            if self.seed and not any(a.mode == MODE_SEED for a in self.attempts):
                self._await_worker()
                seed_code, seed_label = self.seed
                self._set_activity(f"grading the starting kernel ({seed_label})")
                self.log(f"seed: grading starting kernel {seed_label}")
                self.attempts.append(self._submit_attempt(-1, 0, f"starting kernel: {seed_label}", seed_code,
                                                          None, MODE_SEED, "seed"))
                self._save()
                self._wait()
                self._finish_generation(-1, previous_best=None)
                generation = max(generation, 0)
            while True:
                reason = self._stop_reason(generation, started)
                if reason:
                    stop_reason = reason
                    break
                self._await_worker()
                if self._stop_requested:
                    continue  # -> _stop_reason returns "stopped"
                previous_best = self._best_peak()
                parents = self._top_parents(max(1, self.ls.parents_top_k))
                stagnant = self._stagnant_generations(generation)
                explore = self.ls.explore_after > 0 and stagnant >= self.ls.explore_after
                tried = self._tried_approaches() if explore else None
                siblings: list[str] = []
                n = self.ls.candidates_per_generation
                if explore:
                    self.log(f"generation {generation}: EXPLORE (no >= {self.ls.min_rel_improvement:.0%} gain for "
                             f"{stagnant} generations)")
                for i in range(n):
                    if self._stop_requested:
                        break
                    # explore: keep one exploit candidate on the best kernel (if n > 1), the rest try new designs
                    mode = MODE_EXPLORE if explore and (i > 0 or n == 1) else MODE_EXPLOIT
                    parent = parents[0] if explore else parents[i % len(parents)]
                    attempt = self._propose_one(generation, i, n, parent, siblings, mode=mode, tried=tried,
                                                stagnant=stagnant)
                    siblings.append(attempt.hypothesis)
                    self.attempts.append(attempt)
                    self._save()
                self._wait()
                self._finish_generation(generation, previous_best)
                generation += 1
        except WorkerDown as e:
            stop_reason = "worker_down"
            self.log(str(e))
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        finally:
            self._save()
        best = self._best_attempt()
        outcome = LoopOutcome(self.loop_id, stop_reason, generation, len(self.attempts), best)
        self.tracker.summary({"stop_reason": stop_reason, "generations": generation,
                              "best_peak_fraction": best.peak if best else None,
                              "best_candidate": best.candidate_id if best else None,
                              "best_hypothesis": best.hypothesis if best else None})
        self.log(f"loop {self.loop_id} stopped: {stop_reason}; best="
                 f"{f'{best.peak:.4f} ({best.label})' if best else 'none'}")
        return outcome
