"""W&B-backed queue. Every candidate is one W&B run (config.kind == "candidate").

Ordering guarantee: a candidate run is created with status=submitting, its
solution.py is uploaded synchronously through the public API, and only then is
status flipped to pending, so a worker never claims a candidate without code.

Concurrency model: one worker per project. Claiming is read-then-write and is
not atomic across several workers.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import wandb

from looppp import contract as c
from looppp.contract import CandidateRecord, CandidateSpec, GradeResult


class WandbQueue:
    def __init__(self, entity: str, project: str, api_key: str | None = None, timeout: int = 60):
        if not entity:
            raise ValueError("W&B entity is required (set queue.entity or $WANDB_ENTITY)")
        self.entity, self.project, self.api_key = entity, project, api_key
        self.api = wandb.Api(api_key=api_key, timeout=timeout)
        self._worker_run = None

    # --- helpers ----------------------------------------------------------
    @property
    def _project_path(self) -> str:
        return f"{self.entity}/{self.project}"

    def _settings(self) -> wandb.Settings:
        return wandb.Settings(api_key=self.api_key) if self.api_key else wandb.Settings()

    def _run(self, path: str):
        self.api.flush()
        return self.api.run(path)

    @staticmethod
    def _path(run) -> str:
        """Public-API runs expose ``path`` as [entity, project, run_id]."""
        return run.path if isinstance(run.path, str) else "/".join(run.path)

    @classmethod
    def _record(cls, run) -> CandidateRecord:
        summary: dict[str, Any] = dict(run.summary_metrics or {})
        status = summary.get("status", c.STATUS_SUBMITTING)
        result = GradeResult.from_summary(summary) if status in (c.STATUS_DONE, c.STATUS_ERROR) else None
        return CandidateRecord(
            id=cls._path(run),
            spec=CandidateSpec.from_config(dict(run.config)),
            status=status,
            attempts=int(summary.get("attempts", 0)),
            result=result,
        )

    # --- agent side -------------------------------------------------------
    def submit(self, spec: CandidateSpec, code: str) -> str:
        if c.sha256_text(code) != spec.code_sha256:
            raise ValueError("spec.code_sha256 does not match code")
        run = wandb.init(
            entity=self.entity, project=self.project, job_type=c.KIND_CANDIDATE,
            group=spec.loop_id, name=f"{spec.problem}-g{spec.generation:03d}-i{spec.index}",
            config=spec.to_config(), reinit="create_new", settings=self._settings(),
        )
        run.summary["status"] = c.STATUS_SUBMITTING
        path = "/".join([run.entity, run.project, run.id])
        run.finish()

        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / c.SOLUTION_FILENAME
            f.write_text(code)
            public = self._run(path)
            public.upload_file(str(f), root=tmp)
            public.summary.update({"status": c.STATUS_PENDING, "attempts": 0, "submitted_ts": time.time()})
        return path

    def get(self, candidate_id: str) -> CandidateRecord:
        return self._record(self._run(candidate_id))

    def abandon(self, candidate_id: str, reason: str) -> None:
        run = self._run(candidate_id)
        if run.summary_metrics.get("status") not in c.TERMINAL_STATUSES:
            run.summary.update({"status": c.STATUS_ABANDONED, "abandon_reason": reason})

    def list_candidates(self, loop_id: str | None = None, limit: int = 100) -> list[CandidateRecord]:
        filters: dict[str, Any] = {"config.kind": c.KIND_CANDIDATE}
        if loop_id:
            filters["config.loop_id"] = loop_id
        self.api.flush()
        runs = self.api.runs(self._project_path, filters=filters, order="-created_at", per_page=min(limit, 100))
        out = []
        for run in runs:
            out.append(self._record(run))
            if len(out) >= limit:
                break
        return out

    def worker_heartbeat_age(self) -> float | None:
        self.api.flush()
        runs = self.api.runs(self._project_path, filters={"config.kind": c.KIND_WORKER},
                             order="-created_at", per_page=5)
        latest = None
        for run in runs:
            ts = (run.summary_metrics or {}).get("heartbeat_ts")
            if ts is not None:
                latest = max(latest or 0.0, float(ts))
        return None if latest is None else time.time() - latest

    # --- worker side ------------------------------------------------------
    def claim_next(self, worker_id: str) -> tuple[CandidateRecord, str] | None:
        self.api.flush()
        runs = self.api.runs(
            self._project_path,
            filters={"config.kind": c.KIND_CANDIDATE, "summary_metrics.status": c.STATUS_PENDING},
            order="+created_at", per_page=5,
        )
        for run in runs:
            run = self._run(self._path(run))
            if run.summary_metrics.get("status") != c.STATUS_PENDING:
                continue
            run.summary.update({"status": c.STATUS_RUNNING, "worker_id": worker_id, "claimed_ts": time.time()})
            rec = self._record(run)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    run.file(c.SOLUTION_FILENAME).download(root=tmp, replace=True)
                    code = (Path(tmp) / c.SOLUTION_FILENAME).read_text()
            except Exception as e:  # noqa: BLE001 - any download problem is an infra failure
                self.fail(rec.id, c.STAGE_DOWNLOAD, f"{type(e).__name__}: {e}", worker_id)
                continue
            return rec, code
        return None

    def complete(self, candidate_id: str, result: GradeResult) -> None:
        self._run(candidate_id).summary.update({"status": c.STATUS_DONE, "completed_ts": time.time(),
                                                **result.to_summary()})

    def fail(self, candidate_id: str, stage: str, reason: str, worker_id: str = "") -> None:
        result = GradeResult(correct=False, fail_stage=stage, fail_reason=reason, worker_id=worker_id)
        self._run(candidate_id).summary.update({"status": c.STATUS_ERROR, "completed_ts": time.time(),
                                                **result.to_summary()})

    def requeue_orphans(self, max_attempts: int) -> list[str]:
        self.api.flush()
        runs = self.api.runs(
            self._project_path,
            filters={"config.kind": c.KIND_CANDIDATE, "summary_metrics.status": c.STATUS_RUNNING},
            per_page=50,
        )
        touched = []
        for run in runs:
            attempts = int(run.summary_metrics.get("attempts", 0)) + 1
            if attempts >= max_attempts:
                result = GradeResult(correct=False, fail_stage=c.STAGE_WORKER_LOST,
                                     fail_reason=f"worker died {attempts} times while grading this candidate")
                run.summary.update({"status": c.STATUS_ERROR, "attempts": attempts, **result.to_summary()})
            else:
                run.summary.update({"status": c.STATUS_PENDING, "attempts": attempts})
            touched.append(self._path(run))
        return touched

    def _ensure_worker_run(self, worker_id: str):
        if self._worker_run is None:
            self._worker_run = wandb.init(
                entity=self.entity, project=self.project, job_type=c.KIND_WORKER, name=worker_id,
                config={"kind": c.KIND_WORKER, "worker_id": worker_id},
                reinit="create_new", settings=self._settings(),
            )
        return self._worker_run

    def heartbeat(self, worker_id: str, info: dict) -> None:
        run = self._ensure_worker_run(worker_id)
        ts = time.time()
        numeric = {k: v for k, v in info.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        run.log({"heartbeat_ts": ts, **numeric})
        run.summary["heartbeat_ts"] = ts
        for k, v in info.items():
            if k not in numeric:
                run.summary[k] = v

    def worker_log(self, worker_id: str, data: dict) -> None:
        run = self._ensure_worker_run(worker_id)
        for k, v in data.items():
            run.summary[k] = v

    def close(self) -> None:
        if self._worker_run is not None:
            self._worker_run.finish()
            self._worker_run = None
