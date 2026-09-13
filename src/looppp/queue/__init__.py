"""Queue backends. ``make_queue`` picks one from config."""
from __future__ import annotations

from looppp.config import Config
from looppp.queue.base import Queue
from looppp.queue.local import LocalQueue


def make_queue(cfg: Config, api_key: str | None = None, backend: str | None = None) -> Queue:
    backend = backend or cfg.queue.backend
    if backend == "local":
        return LocalQueue(cfg.path(cfg.queue.local_root))
    if backend == "wandb":
        from looppp.queue.wandb_queue import WandbQueue  # imports wandb lazily

        return WandbQueue(cfg.queue.entity or "", cfg.queue.project, api_key=api_key)
    raise ValueError(f"unknown queue backend {backend!r}")


__all__ = ["LocalQueue", "Queue", "make_queue"]
