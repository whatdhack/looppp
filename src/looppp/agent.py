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
from looppp.prompts import HistoryItem, build_propose_messages, build_repair_messages
from looppp.queue.base import Queue
from looppp.tracker import Tracker

# Attempt statuses that never reach the queue.
LOCAL_FAILED = "local_failed"
OPEN_STATUSES = (c.STATUS_PENDING, c.STATUS_RUNNING, c.STATUS_SUBMITTING)


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
                 log: Callable[[str], None] = print):
        self.cfg, self.ls, self.qs = cfg, cfg.loop, cfg.queue
        self.problem, self.queue, self.llm, self.tracker, self.archive = problem, queue, llm, tracker, archive
        self.loop_id, self.deck_commit = loop_id, deck_commit
        self.clock, self.sleep, self.log = clock, sleep, log
        self.dir = cfg.path(self.ls.state_dir) / loop_id
        (self.dir / "code").mkdir(parents=True, exist_ok=True)
        self.attempts: list[Attempt] = []
        self.best_gen = -1
        self._down_since: float | None = None

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
                                   f"{self.loop_id}: no fresh worker heartbeat ({seen}). Restart the molab "
                                   "evaluator notebook; the loop is paused and will resume automatically.")
            if now - self._down_since > self.qs.max_worker_down_hours * 3600:
                raise WorkerDown(f"worker offline for more than {self.qs.max_worker_down_hours} h")
        return alive

    def _await_worker(self) -> None:
        while not self._worker_alive():
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
        rel = f"code/g{generation:03d}-i{index}.py"
        (self.dir / rel).write_text(code)
        return rel

    def _propose_one(self, generation: int, index: int, n: int, parent, siblings: list[str]) -> Attempt:
        parent_code, parent_label, parent_result, parent_id = parent
        messages = build_propose_messages(
            self.problem, parent_code=parent_code, parent_label=parent_label, parent_result=parent_result,
            history=self._history(), failures=self._failure_logs(generation - 1),
            sibling_hypotheses=siblings, index=index, n=n, best_peak=self._best_peak(),
        )
        hypothesis, code, errors = "(no reply)", "", ["no reply"]
        for repair in range(self.ls.precheck_repairs + 1):
            try:
                reply = self.llm.complete(messages)
            except EmptyCompletionError as e:
                return self._local_failure(generation, index, "(model returned nothing usable)", c.STAGE_LLM, str(e), "")
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
            return self._local_failure(generation, index, hypothesis, c.STAGE_PRECHECK, "; ".join(errors), code)

        sha = c.sha256_text(code)
        dup = next((a for a in self.attempts if a.code_sha256 == sha and a.candidate_id), None)
        if dup:
            return self._local_failure(generation, index, hypothesis, c.STAGE_DUPLICATE,
                                       f"identical to {dup.label}", code)

        attempt = Attempt(generation, index, hypothesis, c.STATUS_SUBMITTING, self.llm.model, sha,
                          self._store_code(generation, index, code), parent=parent_id)
        spec = CandidateSpec(problem=self.problem.name, deck_commit=self.deck_commit, loop_id=self.loop_id,
                             generation=generation, index=index, model=self.llm.model, hypothesis=hypothesis,
                             code_sha256=sha, parent=parent_id)
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
        last = self.clock()
        while True:
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
        if improved:
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
            while True:
                reason = self._stop_reason(generation, started)
                if reason:
                    stop_reason = reason
                    break
                self._await_worker()
                previous_best = self._best_peak()
                parent = self._parent()
                siblings: list[str] = []
                n = self.ls.candidates_per_generation
                for i in range(n):
                    attempt = self._propose_one(generation, i, n, parent, siblings)
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
