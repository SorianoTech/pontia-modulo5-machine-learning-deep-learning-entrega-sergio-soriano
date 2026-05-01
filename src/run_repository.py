from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src import config


def _load_registry() -> dict[str, Any]:
    path = Path(config.RUNS_REGISTRY_PATH)
    if not path.exists():
        return {"runs": []}

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_registry(data: dict[str, Any]) -> None:
    path = Path(config.RUNS_REGISTRY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def register_run(payload: dict[str, Any]) -> dict[str, Any]:
    registry = _load_registry()
    run_entry = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    registry["runs"].append(run_entry)
    _save_registry(registry)
    return run_entry


def list_runs(limit: int | None = None) -> list[dict[str, Any]]:
    runs = _load_registry().get("runs", [])
    runs = sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)
    if limit is not None:
        return runs[:limit]
    return runs


def get_run(run_id: str) -> dict[str, Any] | None:
    for run in _load_registry().get("runs", []):
        if run.get("id") == run_id:
            return run
    return None
