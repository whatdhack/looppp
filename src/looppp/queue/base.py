"""Queue interface shared by the agent and the worker."""
from __future__ import annotations

from typing import Protocol

from looppp.contract import CandidateRecord, CandidateSpec, GradeResult


class Queue(Protocol):
    # --- agent side -------------------------------------------------------
    def submit(self, spec: CandidateSpec, code: str) -> str:
        """Store the candidate and mark it pending. Returns the candidate id."""

    def get(self, candidate_id: str) -> CandidateRecord: ...

    def abandon(self, candidate_id: str, reason: str) -> None:
        """Agent gave up waiting. No-op if the candidate is already terminal."""

    def list_candidates(self, loop_id: str | None = None, limit: int = 100) -> list[CandidateRecord]: ...

    def worker_heartbeat_age(self) -> float | None:
        """Seconds since the most recent worker heartbeat, or None if no worker ever reported."""

    # --- worker side ------------------------------------------------------
    def claim_next(self, worker_id: str) -> tuple[CandidateRecord, str] | None:
        """Oldest pending candidate -> running. Returns (record, code) or None."""

    def complete(self, candidate_id: str, result: GradeResult) -> None:
        """Worker finished grading (pass or fail). Sets status=done."""

    def fail(self, candidate_id: str, stage: str, reason: str, worker_id: str = "") -> None:
        """Infrastructure failure. Sets status=error."""

    def requeue_orphans(self, max_attempts: int) -> list[str]:
        """Candidates stuck in running (their worker died): back to pending, or error after max_attempts."""

    def heartbeat(self, worker_id: str, info: dict) -> None: ...

    def worker_log(self, worker_id: str, data: dict) -> None:
        """Attach worker-level facts (calibration, env report) to the worker's record."""

    def close(self) -> None: ...
