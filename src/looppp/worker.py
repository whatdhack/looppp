"""The evaluator loop: claim -> grade -> report, with a heartbeat thread.

Runs anywhere a Grader works: inside the molab notebook (thread) or as
``looppp worker`` on any GPU box.
"""
from __future__ import annotations

import socket
import statistics
import threading
import time
import uuid
from typing import Callable

from looppp import contract as c
from looppp.grade import Grader
from looppp.queue.base import Queue


def new_worker_id(prefix: str = "worker") -> str:
    return f"{prefix}-{socket.gethostname()[:20]}-{uuid.uuid4().hex[:6]}"


class Worker:
    def __init__(self, queue: Queue, grader: Grader, *, poll_seconds: float = 20.0,
                 heartbeat_seconds: float = 30.0, max_attempts: int = 2,
                 log: Callable[[str], None] = print):
        self.queue, self.grader = queue, grader
        self.worker_id = grader.worker_id
        self.poll_seconds, self.heartbeat_seconds = poll_seconds, heartbeat_seconds
        self.max_attempts = max_attempts
        self.log = log
        self.stop_event = threading.Event()
        self.status: dict = {"state": "idle", "graded": 0, "errors": 0, "current": "", "last": None,
                             "started_ts": None, "worker_id": self.worker_id, "error": ""}
        self._thread: threading.Thread | None = None

    # --- heartbeat ----------------------------------------------------------
    def _beat(self) -> None:
        try:
            self.queue.heartbeat(self.worker_id, {"graded": self.status["graded"], "errors": self.status["errors"],
                                                  "state": self.status["state"], "current": self.status["current"]})
        except Exception as e:  # noqa: BLE001 - never let telemetry kill grading
            self.log(f"heartbeat failed: {type(e).__name__}: {e}")

    def _heartbeat_loop(self) -> None:
        while not self.stop_event.wait(self.heartbeat_seconds):
            self._beat()

    # --- main loop ----------------------------------------------------------
    def run(self, max_candidates: int | None = None, idle_exit_seconds: float | None = None) -> dict:
        self.status.update(state="starting", started_ts=time.time(), error="")
        desc = self.grader.describe()
        try:
            self.queue.worker_log(self.worker_id, desc)
            orphans = self.queue.requeue_orphans(self.max_attempts)
        except Exception as e:  # noqa: BLE001 - in a thread nobody would see the traceback
            self.status.update(state="failed", error=f"{type(e).__name__}: {e}")
            self.log(f"worker could not start: {type(e).__name__}: {e}")
            return self.status
        if orphans:
            self.log(f"re-queued / errored {len(orphans)} orphaned candidates")
        self._beat()  # creates the worker record before the heartbeat thread shares it
        hb = threading.Thread(target=self._heartbeat_loop, name="looppp-heartbeat", daemon=True)
        hb.start()

        idle_since = time.monotonic()
        try:
            while not self.stop_event.is_set():
                self.status.update(state="polling", current="")
                try:
                    claimed = self.queue.claim_next(self.worker_id)
                except Exception as e:  # noqa: BLE001
                    self.status["errors"] += 1
                    self.log(f"claim failed: {type(e).__name__}: {e}")
                    self.stop_event.wait(self.poll_seconds)
                    continue
                if claimed is None:
                    if idle_exit_seconds is not None and time.monotonic() - idle_since > idle_exit_seconds:
                        break
                    self.stop_event.wait(self.poll_seconds)
                    continue

                rec, code = claimed
                self.status.update(state="grading", current=rec.id)
                self.log(f"grading {rec.id} ({rec.spec.problem}, gen {rec.spec.generation}.{rec.spec.index})")
                self._grade_one(rec, code, desc)
                idle_since = time.monotonic()
                if max_candidates is not None and self.status["graded"] >= max_candidates:
                    break
        finally:
            self.stop_event.set()
            hb.join(timeout=5)
            self.status.update(state="stopped", current="")
            self._beat()
        return self.status

    def _grade_one(self, rec: c.CandidateRecord, code: str, desc: dict) -> None:
        if desc.get("deck_commit") not in ("fake", rec.spec.deck_commit):
            self.queue.fail(rec.id, c.STAGE_INTEGRITY,
                            f"candidate targets deck {rec.spec.deck_commit}, worker has {desc.get('deck_commit')}",
                            self.worker_id)
            return
        if c.sha256_text(code) != rec.spec.code_sha256:
            self.queue.fail(rec.id, c.STAGE_INTEGRITY, "solution.py sha256 does not match the submitted spec",
                            self.worker_id)
            return
        try:
            result = self.grader.grade(rec.spec.problem, code)
        except Exception as e:  # noqa: BLE001
            self.status["errors"] += 1
            self.log(f"grader crashed on {rec.id}: {type(e).__name__}: {e}")
            self.queue.fail(rec.id, c.STAGE_WORKER_EXCEPTION, f"{type(e).__name__}: {e}", self.worker_id)
            return
        self.queue.complete(rec.id, result)
        self.status["graded"] += 1
        self.status["last"] = {"id": rec.id, "correct": result.correct, "peak_fraction": result.peak_fraction,
                               "fail_stage": result.fail_stage, "seconds": result.grade_seconds}
        self.log(f"  -> correct={result.correct} peak_fraction={result.peak_fraction} "
                 f"fail_stage={result.fail_stage} ({result.grade_seconds}s)")

    # --- thread helpers (notebook) -----------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_thread(self, **run_kwargs) -> threading.Thread:
        if self._thread and self._thread.is_alive():
            return self._thread
        self.stop_event.clear()
        self._thread = threading.Thread(target=self.run, kwargs=run_kwargs, name="looppp-worker", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self, join_timeout: float | None = None) -> None:
        self.stop_event.set()
        if self._thread and join_timeout is not None:
            self._thread.join(join_timeout)


def calibrate(grader: Grader, problem: str, code: str, runs: int = 3,
              published_peak_fraction: float | None = None) -> dict:
    """Grade a known published solution several times on this GPU."""
    results = [grader.grade(problem, code) for _ in range(runs)]
    scores = [r.peak_fraction for r in results if r.scored]
    report = {
        "problem": problem, "runs": runs, "scored": len(scores),
        "fail_stages": [r.fail_stage for r in results if not r.scored],
        "published": published_peak_fraction,
    }
    if scores:
        report.update(mean=round(statistics.fmean(scores), 4), min=min(scores), max=max(scores))
        if published_peak_fraction:
            report["ratio_to_published"] = round(report["mean"] / published_peak_fraction, 3)
    else:
        first = results[0] if results else None
        report["first_failure"] = (first.fail_reason, first.check_tail[-1500:]) if first else None
    return report
