"""File-system queue: same contract as W&B, for tests and single-machine runs.

Layout::

    <root>/.lock
    <root>/candidates/<id>/spec.json      immutable
    <root>/candidates/<id>/solution.py    immutable
    <root>/candidates/<id>/state.json     status, attempts, result, timestamps
    <root>/worker.json                    latest heartbeat
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import time
import uuid
from pathlib import Path

from looppp import contract as c
from looppp.contract import CandidateRecord, CandidateSpec, GradeResult


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=1, default=str))
    os.replace(tmp, path)


class LocalQueue:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "candidates").mkdir(parents=True, exist_ok=True)
        (self.root / ".lock").touch(exist_ok=True)

    @contextlib.contextmanager
    def _locked(self):
        with open(self.root / ".lock", "r+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def _dir(self, cid: str) -> Path:
        return self.root / "candidates" / cid

    def _state(self, cid: str) -> dict:
        return json.loads((self._dir(cid) / "state.json").read_text())

    def _set_state(self, cid: str, **updates) -> dict:
        state = {**self._state(cid), **updates, "updated_ts": time.time()}
        _write_json(self._dir(cid) / "state.json", state)
        return state

    def _record(self, cid: str) -> CandidateRecord:
        d = self._dir(cid)
        spec = CandidateSpec.from_config(json.loads((d / "spec.json").read_text()))
        st = self._state(cid)
        result = GradeResult.from_summary(st["result"]) if st.get("result") else None
        return CandidateRecord(cid, spec, st["status"], st.get("attempts", 0), result)

    # --- agent side -------------------------------------------------------
    def submit(self, spec: CandidateSpec, code: str) -> str:
        if c.sha256_text(code) != spec.code_sha256:
            raise ValueError("spec.code_sha256 does not match code")
        cid = f"{spec.loop_id}-g{spec.generation:03d}-i{spec.index}-{uuid.uuid4().hex[:6]}"
        d = self._dir(cid)
        d.mkdir(parents=True)
        _write_json(d / "spec.json", spec.to_config())
        (d / c.SOLUTION_FILENAME).write_text(code)
        now = time.time()
        with self._locked():
            _write_json(d / "state.json", {"status": c.STATUS_PENDING, "attempts": 0,
                                           "submitted_ts": now, "updated_ts": now})
        return cid

    def get(self, candidate_id: str) -> CandidateRecord:
        with self._locked():
            return self._record(candidate_id)

    def abandon(self, candidate_id: str, reason: str) -> None:
        with self._locked():
            if self._state(candidate_id)["status"] not in c.TERMINAL_STATUSES:
                self._set_state(candidate_id, status=c.STATUS_ABANDONED, abandon_reason=reason)

    def list_candidates(self, loop_id: str | None = None, limit: int = 100) -> list[CandidateRecord]:
        with self._locked():
            ids = sorted((p.name for p in (self.root / "candidates").iterdir()),
                         key=lambda cid: self._state(cid).get("submitted_ts", 0), reverse=True)
            recs = [self._record(cid) for cid in ids]
        if loop_id:
            recs = [r for r in recs if r.spec.loop_id == loop_id]
        return recs[:limit]

    def worker_heartbeat_age(self) -> float | None:
        f = self.root / "worker.json"
        if not f.exists():
            return None
        return time.time() - json.loads(f.read_text())["ts"]

    # --- worker side ------------------------------------------------------
    def claim_next(self, worker_id: str) -> tuple[CandidateRecord, str] | None:
        with self._locked():
            pending = []
            for p in (self.root / "candidates").iterdir():
                st = self._state(p.name)
                if st["status"] == c.STATUS_PENDING:
                    pending.append((st.get("submitted_ts", 0), p.name))
            if not pending:
                return None
            _, cid = min(pending)
            self._set_state(cid, status=c.STATUS_RUNNING, worker_id=worker_id, claimed_ts=time.time())
            rec = self._record(cid)
        return rec, (self._dir(cid) / c.SOLUTION_FILENAME).read_text()

    def complete(self, candidate_id: str, result: GradeResult) -> None:
        with self._locked():
            self._set_state(candidate_id, status=c.STATUS_DONE, result=result.to_summary())

    def fail(self, candidate_id: str, stage: str, reason: str, worker_id: str = "") -> None:
        result = GradeResult(correct=False, fail_stage=stage, fail_reason=reason, worker_id=worker_id)
        with self._locked():
            self._set_state(candidate_id, status=c.STATUS_ERROR, result=result.to_summary())

    def requeue_orphans(self, max_attempts: int) -> list[str]:
        touched = []
        with self._locked():
            for p in (self.root / "candidates").iterdir():
                st = self._state(p.name)
                if st["status"] != c.STATUS_RUNNING:
                    continue
                attempts = st.get("attempts", 0) + 1
                if attempts >= max_attempts:
                    result = GradeResult(correct=False, fail_stage=c.STAGE_WORKER_LOST,
                                         fail_reason=f"worker died {attempts} times while grading this candidate")
                    self._set_state(p.name, status=c.STATUS_ERROR, attempts=attempts, result=result.to_summary())
                else:
                    self._set_state(p.name, status=c.STATUS_PENDING, attempts=attempts)
                touched.append(p.name)
        return touched

    def heartbeat(self, worker_id: str, info: dict) -> None:
        _write_json(self.root / "worker.json", {"worker_id": worker_id, "ts": time.time(), **info})

    def worker_log(self, worker_id: str, data: dict) -> None:
        with (self.root / "worker_log.jsonl").open("a") as f:
            f.write(json.dumps({"worker_id": worker_id, "ts": time.time(), **data}, default=str) + "\n")

    def close(self) -> None:
        pass
