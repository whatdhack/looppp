"""Loop-level telemetry: one W&B "agent" run per loop (generation metrics, alerts)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    def log(self, data: dict) -> None: ...

    def summary(self, data: dict) -> None: ...

    def alert(self, title: str, text: str) -> None: ...

    def finish(self) -> None: ...


class ConsoleTracker:
    def __init__(self, path: Path | None = None, echo: bool = True):
        self.path, self.echo = path, echo
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.alerts: list[tuple[str, str]] = []

    def _write(self, kind: str, data: dict) -> None:
        if self.path:
            with self.path.open("a") as f:
                f.write(json.dumps({"kind": kind, "ts": time.time(), **data}, default=str) + "\n")

    def log(self, data: dict) -> None:
        self._write("log", data)
        if self.echo:
            print("[looppp]", {k: v for k, v in data.items() if not isinstance(v, str) or len(v) < 120})

    def summary(self, data: dict) -> None:
        self._write("summary", data)

    def alert(self, title: str, text: str) -> None:
        self.alerts.append((title, text))
        self._write("alert", {"title": title, "text": text})
        if self.echo:
            print(f"[looppp ALERT] {title}: {text}")

    def finish(self) -> None:
        pass


class TeeTracker:
    """Send everything to several trackers; a failing secondary never breaks the loop."""

    def __init__(self, primary: Tracker, *others: Tracker):
        self.primary, self.others = primary, others

    @property
    def alerts(self):
        return getattr(self.primary, "alerts", [])

    def _each(self, method: str, *args) -> None:
        getattr(self.primary, method)(*args)
        for t in self.others:
            try:
                getattr(t, method)(*args)
            except Exception as e:  # noqa: BLE001
                if hasattr(self.primary, "log") and method != "log":
                    self.primary.log({"tracker_error": f"{type(t).__name__}.{method}: {type(e).__name__}: {e}"})

    def log(self, data: dict) -> None:
        self._each("log", data)

    def summary(self, data: dict) -> None:
        self._each("summary", data)

    def alert(self, title: str, text: str) -> None:
        self._each("alert", title, text)

    def finish(self) -> None:
        self._each("finish")


class WandbTracker:
    def __init__(self, entity: str, project: str, loop_id: str, config: dict, api_key: str | None = None):
        import wandb

        self._run = wandb.init(
            entity=entity, project=project, job_type="agent", group=loop_id, name=loop_id,
            id=loop_id.replace("/", "-")[:64], resume="allow",
            config={"kind": "agent", "loop_id": loop_id, **config}, reinit="create_new",
            settings=wandb.Settings(api_key=api_key) if api_key else wandb.Settings(),
        )

    def log(self, data: dict) -> None:
        self._run.log(data)

    def summary(self, data: dict) -> None:
        for k, v in data.items():
            self._run.summary[k] = v

    def alert(self, title: str, text: str) -> None:
        try:
            self._run.alert(title=title, text=text)
        except Exception as e:  # noqa: BLE001 - alerts are best effort
            print(f"[looppp ALERT] {title}: {text} (wandb alert failed: {e})")

    def finish(self) -> None:
        self._run.finish()
