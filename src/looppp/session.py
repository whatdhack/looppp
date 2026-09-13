"""Run the whole loop in one process: agent thread + grader thread + local queue.

This is what the single-notebook mode (notebooks/loop.py) uses on the GPU box. The
agent and the worker are the same classes as the distributed mode; they simply talk
through a LocalQueue on disk instead of W&B, so there is no cross-machine heartbeat.
"""
from __future__ import annotations

import collections
import copy
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from looppp import contract as c
from looppp.agent import AgentLoop, LoopOutcome
from looppp.archive import Archive
from looppp.config import Config
from looppp.grade import Grader
from looppp.llm import LLM
from looppp.problems import Problem
from looppp.queue.local import LocalQueue
from looppp.tracker import ConsoleTracker, Tracker
from looppp.worker import Worker


def sparkline_svg(values: list[float | None], width: int = 560, height: int = 110) -> str:
    """Dependency-free step chart of best-so-far peak_fraction per attempt (None = nothing scored yet)."""
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if not pts:
        return f'<svg width="{width}" height="{height}"><text x="8" y="{height // 2}" font-size="13" fill="#888">no scored candidates yet</text></svg>'
    pad, n = 28, max(len(values) - 1, 1)
    lo, hi = 0.0, max(v for _, v in pts) * 1.1 or 1.0
    x = lambda i: pad + (width - 2 * pad) * i / n
    y = lambda v: height - pad - (height - 2 * pad) * (v - lo) / (hi - lo)
    path = " ".join(f"{'M' if k == 0 else 'L'}{x(i):.1f},{y(v):.1f}" for k, (i, v) in enumerate(pts))
    dots = "".join(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="2.5" fill="#4c78a8"/>' for i, v in pts)
    last_i, last_v = pts[-1]
    return (f'<svg width="{width}" height="{height}" font-family="sans-serif" font-size="11">'
            f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#bbb"/>'
            f'<path d="{path}" fill="none" stroke="#4c78a8" stroke-width="2"/>{dots}'
            f'<text x="{pad}" y="14" fill="#666">best peak_fraction so far</text>'
            f'<text x="{x(last_i) - 4:.1f}" y="{y(last_v) - 6:.1f}" text-anchor="end" fill="#4c78a8">{last_v:.4f}</text>'
            f'<text x="{pad}" y="{height - 8}" fill="#888">attempt 1</text>'
            f'<text x="{width - pad}" y="{height - 8}" text-anchor="end" fill="#888">attempt {len(values)}</text></svg>')


@dataclass
class SessionOptions:
    candidates_per_generation: int = 3
    max_generations: int = 20
    patience: int = 5
    wall_budget_hours: float = 11.0          # molab stops sessions at 12 h
    target_peak_fraction: float | None = None
    poll_seconds: float = 5.0


class LoopSession:
    def __init__(self, cfg: Config, problem: Problem, llm: LLM, grader: Grader, loop_id: str, deck_commit: str,
                 options: SessionOptions, *, tracker: Tracker | None = None, archive: Archive | None = None,
                 state_root: Path | None = None, log_lines: int = 400):
        self.cfg = copy.deepcopy(cfg)
        ls, qs = self.cfg.loop, self.cfg.queue
        ls.problem, ls.model = problem.name, llm.model
        ls.candidates_per_generation = options.candidates_per_generation
        ls.max_generations, ls.patience = options.max_generations, options.patience
        ls.wall_budget_hours, ls.target_peak_fraction = options.wall_budget_hours, options.target_peak_fraction
        qs.poll_seconds = options.poll_seconds
        qs.heartbeat_stale_seconds = max(60.0, 10 * options.poll_seconds)
        qs.max_worker_down_hours = 0.05  # the grader thread lives in this process: silence means it died

        root = Path(state_root) if state_root else self.cfg.path(".looppp/session")
        self.cfg.loop.state_dir = str(root / "loops")
        self.problem, self.llm, self.grader, self.loop_id = problem, llm, grader, loop_id
        self.logs: collections.deque[str] = collections.deque(maxlen=log_lines)
        self.tracker = tracker or ConsoleTracker(root / "loops" / loop_id / "events.jsonl", echo=False)
        self.archive = archive or Archive(self.cfg.path(self.cfg.archive.dir))
        self.queue = LocalQueue(root / "queue")
        self._stop = threading.Event()
        self.worker = Worker(self.queue, grader, poll_seconds=min(options.poll_seconds, 5.0),
                             heartbeat_seconds=min(options.poll_seconds, 10.0),
                             max_attempts=self.cfg.queue.max_attempts, log=lambda s: self._log(f"[grader] {s}"))
        self.loop = AgentLoop(self.cfg, problem, self.queue, llm, self.tracker, self.archive, loop_id, deck_commit,
                              sleep=self._sleep, log=lambda s: self._log(f"[agent] {s}"))
        self.state = "created"
        self.error = ""
        self.outcome: LoopOutcome | None = None
        self.started_ts: float | None = None
        self._thread: threading.Thread | None = None

    # --- plumbing -----------------------------------------------------------------------
    def _log(self, line: str) -> None:
        self.logs.append(f"{time.strftime('%H:%M:%S')} {line}")

    def _sleep(self, seconds: float) -> None:
        self._stop.wait(seconds)

    def _run(self) -> None:
        self.state = "running"
        try:
            self.outcome = self.loop.run()
            self.state = "finished"
        except Exception as e:  # noqa: BLE001 - surface it in the notebook instead of a dead thread
            self.error = f"{type(e).__name__}: {e}"
            self.state = "failed"
            self._log(f"[session] loop crashed: {self.error}")
        finally:
            self.worker.stop()  # a grading in progress completes; nothing new is claimed
            try:
                self.tracker.finish()
            except Exception:  # noqa: BLE001
                pass

    # --- public API ---------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self.started_ts = time.time()
        self.worker.start_thread()
        deadline = time.monotonic() + 30  # let the grader report once so the agent never sees a "dead" worker
        while self.queue.worker_heartbeat_age() is None and time.monotonic() < deadline and self.worker.is_running:
            time.sleep(0.05)
        self._thread = threading.Thread(target=self._run, name="looppp-session", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop after the current model call / grading; the running grading subprocess is not killed."""
        self.state = "stopping" if self.is_running else self.state
        self.loop.request_stop()
        self._stop.set()

    def best(self) -> tuple[str, dict] | None:
        own = self.loop._best_attempt()
        if own is None:
            return None
        return self.loop._code(own), {"peak_fraction": own.peak, "label": own.label, "hypothesis": own.hypothesis,
                                      "shape_fractions": own.grade.shape_fractions if own.grade else []}

    def snapshot(self) -> dict:
        attempts = list(self.loop.attempts)
        rows = []
        for a in attempts:
            g = a.grade
            rows.append({
                "gen": f"{a.generation}.{a.index}", "status": a.status,
                "peak_fraction": g.peak_fraction if g and g.scored else None,
                "fail_stage": (g.fail_stage or "") if g else "",
                "hypothesis": a.hypothesis[:120],
                "reason": (g.fail_reason[:160] if g and not g.scored else ""),
            })
        scored = [a for a in attempts if a.peak is not None]
        best = max(scored, key=lambda a: a.peak) if scored else None
        best_so_far, running_best = [], None
        for a in attempts:
            if a.peak is not None and (running_best is None or a.peak > running_best):
                running_best = a.peak
            best_so_far.append(running_best)
        return {
            "state": self.state, "error": self.error, "loop_id": self.loop_id,
            "elapsed_min": round((time.time() - self.started_ts) / 60, 1) if self.started_ts else 0.0,
            "generation": max((a.generation for a in attempts), default=-1),
            "attempts": len(attempts), "graded": sum(1 for a in attempts if a.status == c.STATUS_DONE),
            "pending": sum(1 for a in attempts if a.status in (c.STATUS_PENDING, c.STATUS_RUNNING)),
            "best_peak_fraction": best.peak if best else None,
            "best_label": best.label if best else "",
            "best_so_far": best_so_far,
            "worker": dict(self.worker.status),
            "rows": rows,
            "alerts": list(getattr(self.tracker, "alerts", [])),
            "logs": list(self.logs),
            "stop_reason": self.outcome.stop_reason if self.outcome else "",
            "activity": self.loop.activity if self.is_running else "",
            "activity_seconds": round(time.time() - self.loop.activity_since) if self.is_running else 0,
        }
