"""Shared executor for the per-domain demo SQLite databases (bank, ecomm).

Each domain database is re-seeded from scratch before every statement, then the
statement runs exactly as written. There is deliberately no statement allow-list:
Galileo Agent Control is the only thing that may refuse a query, so an unguarded
demo shows the real consequence of a destructive statement, and re-seeding lets
the next turn start clean.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

_lock = threading.Lock()


def run_on_fresh_db(
    db_path: Path, seed: Callable[[sqlite3.Connection], None], sql: str
) -> str:
    """Re-seed ``db_path`` with ``seed`` then execute ``sql``.

    Returns ``{"success", "row_count", "data": [...]}`` as JSON for reads, or
    ``{"success", "rows_affected", "data": []}`` for writes.
    """
    statement = sql.strip().rstrip(";").strip()
    with _lock:
        conn = sqlite3.connect(db_path)
        try:
            seed(conn)
            conn.commit()
            cur = conn.cursor()
            cur.execute(statement)
            if cur.description is None:
                conn.commit()
                return json.dumps(
                    {"success": True, "rows_affected": cur.rowcount, "data": []}
                )
            cols = [d[0] for d in cur.description]
            data = [dict(zip(cols, row)) for row in cur.fetchall()]
            return json.dumps({"success": True, "row_count": len(data), "data": data})
        except sqlite3.Error as exc:
            return json.dumps(
                {"success": False, "error": f"SQL error: {exc}", "row_count": 0, "data": []}
            )
        finally:
            conn.close()
