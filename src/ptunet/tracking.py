"""Experiment tracking with a local JSONL record and optional Weights & Biases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol


class Tracker(Protocol):
    """Small tracking interface used by the trainer."""

    def log(self, metrics: dict[str, Any], step: int) -> None: ...

    def update_summary(self, values: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullTracker:
    """Tracker used when metric persistence is intentionally disabled."""

    def log(self, metrics: dict[str, Any], step: int) -> None:
        del metrics, step

    def update_summary(self, values: dict[str, Any]) -> None:
        del values

    def close(self) -> None:
        return None


class JsonlTracker:
    """Append metrics to a portable JSON Lines file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.path.with_name("summary.json")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        payload = {"step": int(step), **metrics}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def update_summary(self, values: dict[str, Any]) -> None:
        current: dict[str, Any] = {}
        if self.summary_path.exists():
            current = json.loads(self.summary_path.read_text(encoding="utf-8"))
        current.update(values)
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(current, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.summary_path)

    def close(self) -> None:
        return None


class WandbTracker:
    """Mirror metrics to a W&B run while retaining the same tracker interface."""

    def __init__(
        self,
        project: str,
        name: str | None,
        config: dict[str, Any],
        mode: Literal["online", "offline", "disabled", "shared"] = "online",
        entity: str | None = None,
        tags: tuple[str, ...] = (),
    ) -> None:
        try:
            import wandb
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "W&B tracking requires `python -m pip install -e '.[tracking]'`"
            ) from error
        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            name=name,
            config=config,
            mode=mode,
            entity=entity,
            tags=list(tags),
        )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        if self._run is not None:
            for key, value in values.items():
                self._run.summary[key] = value

    def close(self) -> None:
        if self._run is not None:
            self._run.finish()


class CompositeTracker:
    """Send the same metric event to multiple trackers."""

    def __init__(self, trackers: list[Tracker]) -> None:
        self.trackers = trackers

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for tracker in self.trackers:
            tracker.log(metrics, step)

    def update_summary(self, values: dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.update_summary(values)

    def close(self) -> None:
        for tracker in reversed(self.trackers):
            tracker.close()


__all__ = [
    "CompositeTracker",
    "JsonlTracker",
    "NullTracker",
    "Tracker",
    "WandbTracker",
]
