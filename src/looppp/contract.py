"""The shared schema between the agent (WSL) and the worker (molab).

Both sides import this module and nothing else from each other, so a renamed
field breaks tests instead of silently stalling the queue.

Lifecycle of a candidate:

    pending --(worker claims)--> running --> done
       |                            '-----> error      (infra: download, worker crash, retries exhausted)
       '--(agent gives up)--> abandoned

A candidate left in ``running`` by a worker that died is re-queued by the next
worker (``attempts`` + 1) and becomes ``error`` / ``worker_lost`` after
``max_attempts``.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 1

# W&B config["kind"] values (filterable via "config.kind").
KIND_CANDIDATE = "candidate"
KIND_WORKER = "worker"
KIND_AGENT = "agent"

# summary["status"] values.
STATUS_SUBMITTING = "submitting"
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_ABANDONED = "abandoned"
TERMINAL_STATUSES = frozenset({STATUS_DONE, STATUS_ERROR, STATUS_ABANDONED})

# GradeResult.fail_stage values.
STAGE_LLM = "llm"                    # model returned nothing usable (agent side)
STAGE_SUBMIT = "submit"              # agent could not submit to the queue
STAGE_PRECHECK = "precheck"          # rejected locally on WSL, never submitted
STAGE_DUPLICATE = "duplicate"        # identical code already graded
STAGE_DOWNLOAD = "download"          # worker could not fetch solution.py
STAGE_INTEGRITY = "integrity"        # sha256 or deck commit mismatch
STAGE_IMPORT = "import"              # solution.py failed to import / compile
STAGE_FORBIDDEN = "forbidden"        # check.py found a forbidden op
STAGE_CHECK = "check"                # correctness failure
STAGE_BENCHMARK = "benchmark"        # benchmark.py failed or produced no score
STAGE_TIMEOUT = "timeout"
STAGE_TOOLCHAIN = "toolchain"        # needed a compiler (nvcc) the worker does not have
STAGE_TAMPER = "tamper"              # grader files changed during grading
STAGE_WORKER_LOST = "worker_lost"
STAGE_WORKER_EXCEPTION = "worker_exception"

SOLUTION_FILENAME = "solution.py"
LOG_TAIL_CHARS = 6000


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tail(text: str, n: int = LOG_TAIL_CHARS) -> str:
    return text if len(text) <= n else "...[truncated]...\n" + text[-n:]


@dataclass
class CandidateSpec:
    """Written once by the agent at submission; never modified afterwards."""

    problem: str
    deck_commit: str
    loop_id: str
    generation: int
    index: int
    model: str
    hypothesis: str
    code_sha256: str
    parent: str | None = None
    schema_version: int = SCHEMA_VERSION

    def to_config(self) -> dict[str, Any]:
        return {"kind": KIND_CANDIDATE, **asdict(self)}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "CandidateSpec":
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: cfg[k] for k in names if k in cfg})


@dataclass
class GradeResult:
    """Written by the worker (or by the agent for precheck/duplicate rejections)."""

    correct: bool
    peak_fraction: float | None = None
    shape_fractions: list[float] = field(default_factory=list)
    fail_stage: str | None = None
    fail_reason: str = ""
    check_tail: str = ""
    bench_tail: str = ""
    grade_seconds: float = 0.0
    gpu_name: str = ""
    torch_version: str = ""
    worker_id: str = ""

    @property
    def scored(self) -> bool:
        return self.correct and self.peak_fraction is not None

    def to_summary(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> "GradeResult":
        names = cls.__dataclass_fields__.keys()
        data = {k: summary[k] for k in names if k in summary}
        data.setdefault("correct", False)
        return cls(**data)


@dataclass
class CandidateRecord:
    """What the queue returns when asked about a candidate."""

    id: str
    spec: CandidateSpec
    status: str
    attempts: int = 0
    result: GradeResult | None = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES
