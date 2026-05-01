from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from src import config


def _db_path() -> Path:
    path = Path(config.RUNS_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_schema() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _migrate_legacy_json_once() -> None:
    legacy_path = Path(config.LEGACY_RUNS_REGISTRY_PATH)
    if not legacy_path.exists():
        return

    try:
        with legacy_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return

    runs = raw.get("runs", []) if isinstance(raw, dict) else []
    if not runs:
        return

    _ensure_schema()
    with _connect() as conn:
        for run in runs:
            run_id = run.get("id", str(uuid4()))
            created_at = run.get("created_at", datetime.now(timezone.utc).isoformat())
            payload = dict(run)
            payload.pop("id", None)
            payload.pop("created_at", None)
            conn.execute(
                "INSERT OR IGNORE INTO runs (id, created_at, payload_json) VALUES (?, ?, ?)",
                (run_id, created_at, json.dumps(payload, ensure_ascii=False)),
            )
        conn.commit()

    legacy_path.rename(legacy_path.with_suffix(".migrated.json"))


def _row_to_run(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"])
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        **payload,
    }


def register_run(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_schema()
    _migrate_legacy_json_once()

    run_entry = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }

    stored_payload = dict(run_entry)
    run_id = stored_payload.pop("id")
    created_at = stored_payload.pop("created_at")

    with _connect() as conn:
        conn.execute(
            "INSERT INTO runs (id, created_at, payload_json) VALUES (?, ?, ?)",
            (run_id, created_at, json.dumps(stored_payload, ensure_ascii=False)),
        )
        conn.commit()

    return run_entry


def list_runs(limit: int | None = None) -> list[dict[str, Any]]:
    _ensure_schema()
    _migrate_legacy_json_once()

    sql = "SELECT id, created_at, payload_json FROM runs ORDER BY created_at DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_run(row) for row in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    _ensure_schema()
    _migrate_legacy_json_once()

    with _connect() as conn:
        row = conn.execute(
            "SELECT id, created_at, payload_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()

    if not row:
        return None
    return _row_to_run(row)
