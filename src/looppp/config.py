"""Load configs/run.yaml + configs/models.yaml into typed settings."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def repo_root(start: Path | None = None) -> Path:
    """Walk up from *start* (default: this file) to the directory holding configs/run.yaml."""
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / "configs" / "run.yaml").is_file():
            return p
    return Path.cwd()


@dataclass
class LoopSettings:
    problem: str = "07_w4a16_gemm"
    model: str = "zai-org/GLM-5.2"
    candidates_per_generation: int = 3
    max_generations: int = 20
    patience: int = 5
    wall_budget_hours: float = 24.0
    target_peak_fraction: float | None = None
    history_window: int = 12
    precheck_repairs: int = 1
    state_dir: str = ".looppp/loops"


@dataclass
class QueueSettings:
    backend: str = "wandb"                  # wandb | local
    entity: str | None = None
    project: str = "looppp"
    local_root: str = ".looppp/queue"
    poll_seconds: float = 30.0
    wait_timeout_hours: float = 3.0         # counted only while the worker is alive
    max_worker_down_hours: float = 14.0     # molab restarts are manual; stop the loop after this
    heartbeat_stale_seconds: float = 300.0
    max_attempts: int = 2


@dataclass
class DeckSettings:
    repo: str = "https://github.com/Infatoshi/kernelbench.com.git"
    commit: str = "62e4c346f5f351e06169676c605267d0291acaa5"
    subdir: str = "benchmarks/hard"
    problems_dir: str = "problems-rtxpro6000"
    local_path: str = ".looppp/deck"


@dataclass
class ArchiveSettings:
    dir: str = "archive"
    git_commit: bool = False


@dataclass
class WorkerSettings:
    expected_gpu: str | None = "RTX PRO 6000"
    check_timeout_seconds: float = 1800.0
    bench_timeout_seconds: float = 1800.0
    poll_seconds: float = 20.0
    heartbeat_seconds: float = 30.0


@dataclass
class CalibrationTarget:
    run_id: str
    published_peak_fraction: float


@dataclass
class ModelSettings:
    max_tokens: int = 32000
    max_tokens_cap: int = 96000
    temperature: float = 0.7
    timeout_seconds: float = 900.0
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    root: Path
    loop: LoopSettings
    queue: QueueSettings
    deck: DeckSettings
    archive: ArchiveSettings
    worker: WorkerSettings
    calibration: dict[str, CalibrationTarget]
    models: dict[str, ModelSettings]
    model_defaults: ModelSettings

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def model_settings(self, model: str) -> ModelSettings:
        return self.models.get(model, self.model_defaults)

    @property
    def deck_root(self) -> Path:
        """Directory containing src/ and the problems dir (benchmarks/hard in the clone)."""
        return self.path(self.deck.local_path) / self.deck.subdir

    @property
    def problems_root(self) -> Path:
        return self.deck_root / self.deck.problems_dir


def _build(cls, data: dict[str, Any] | None):
    data = data or {}
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)


def load_config(root: Path | None = None) -> Config:
    root = root or repo_root()
    run = yaml.safe_load((root / "configs" / "run.yaml").read_text()) or {}
    models_file = root / "configs" / "models.yaml"
    models_raw = yaml.safe_load(models_file.read_text()) if models_file.exists() else {}
    models_raw = models_raw or {}

    defaults = _build(ModelSettings, models_raw.get("defaults"))
    models = {}
    for name, overrides in (models_raw.get("models") or {}).items():
        merged = {**defaults.__dict__, **(overrides or {})}
        models[name] = _build(ModelSettings, merged)

    queue = _build(QueueSettings, run.get("queue"))
    queue.entity = queue.entity or os.environ.get("WANDB_ENTITY")
    queue.project = os.environ.get("LOOPPP_PROJECT", queue.project)

    return Config(
        root=root,
        loop=_build(LoopSettings, run.get("loop")),
        queue=queue,
        deck=_build(DeckSettings, run.get("deck")),
        archive=_build(ArchiveSettings, run.get("archive")),
        worker=_build(WorkerSettings, run.get("worker")),
        calibration={k: _build(CalibrationTarget, v) for k, v in (run.get("calibration") or {}).items()},
        models=models,
        model_defaults=defaults,
    )
