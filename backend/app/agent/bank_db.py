"""Tiny SQLite bank database backing the bank-domain query tool (demo only).

Seeded in the container's ephemeral storage on first use. Two tables:
``customers`` (has name/email, so it drives the PII-control demo) and
``transactions`` (a deliberately PII-free ledger, so the SQL row-limit steer can
be demoed without an output-PII control withholding the answer). Fake data only.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

_DB_PATH = Path("/tmp/bank_customers.db")
_lock = threading.Lock()
_ready = False

# (id, name, email, account_type, balance) — fabricated demo data only.
_CUSTOMERS = [
    ("1001", "Alice Chen", "alice.chen@example.com", "Savings", 4200.75),
    ("1002", "Bob Smith", "bob.smith@example.com", "Checking", 1800.00),
    ("1003", "Carol White", "carol.white@example.com", "Savings", 9100.50),
    ("1004", "David Lee", "david.lee@example.com", "Checking", 540.25),
    ("1005", "Emma Davis", "emma.davis@example.com", "Premier", 15200.00),
    ("1006", "Frank Moore", "frank.moore@example.com", "Savings", 320.00),
]

# Business merchants only — no personal names, emails, or phone numbers, so a
# ledger answer carries no PII for an output-PII control to trigger on.
_MERCHANTS = [
    ("Northwind Grocers", "Groceries", 82.40),
    ("Contoso Fuel", "Transport", 45.10),
    ("Fabrikam Utilities", "Utilities", 130.00),
    ("Tailspin Airlines", "Travel", 615.30),
    ("Adventure Works Cafe", "Dining", 18.75),
    ("Litware Online", "Shopping", 249.99),
    ("Proseware Insurance", "Insurance", 96.20),
]


def _seed_transactions() -> list[tuple[str, str, str, str, str, float]]:
    """42 deterministic ledger rows (7 per account) — enough that a LIMIT matters."""
    rows = []
    for a_idx, customer in enumerate(_CUSTOMERS):
        account_id = customer[0]
        for m_idx, (merchant, category, base) in enumerate(_MERCHANTS):
            rows.append(
                (
                    f"T{9000 + a_idx * len(_MERCHANTS) + m_idx}",
                    account_id,
                    f"2026-08-{1 + (a_idx * 7 + m_idx) % 28:02d}",
                    merchant,
                    category,
                    round(base + a_idx * 3.25, 2),
                )
            )
    return rows


def _ensure_db() -> None:
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        conn = sqlite3.connect(_DB_PATH)
        try:
            conn.executescript(
                """
                DROP TABLE IF EXISTS customers;
                CREATE TABLE customers (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    account_type TEXT NOT NULL,
                    balance REAL NOT NULL
                );
                DROP TABLE IF EXISTS transactions;
                CREATE TABLE transactions (
                    id TEXT PRIMARY KEY,
                    account_id TEXT NOT NULL,
                    txn_date TEXT NOT NULL,
                    merchant TEXT NOT NULL,
                    category TEXT NOT NULL,
                    amount REAL NOT NULL
                );
                """
            )
            conn.executemany(
                "INSERT INTO customers (id, name, email, account_type, balance) "
                "VALUES (?, ?, ?, ?, ?)",
                _CUSTOMERS,
            )
            conn.executemany(
                "INSERT INTO transactions "
                "(id, account_id, txn_date, merchant, category, amount) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                _seed_transactions(),
            )
            conn.commit()
        finally:
            conn.close()
        _ready = True


def query_customers(sql: str) -> str:
    """Execute a read-only ``SELECT`` against the demo bank tables.

    Non-SELECT statements are refused so the demo DB can't be mutated; that guard
    is a safety net independent of any Agent Control policy on this step.
    Returns a JSON string: ``{"success", "row_count", "data": [...]}``.
    """
    _ensure_db()
    statement = sql.strip().rstrip(";").strip()
    if not statement.lower().startswith("select"):
        return json.dumps(
            {"success": False, "error": "Only SELECT queries are allowed.",
             "row_count": 0, "data": []}
        )
    with _lock:
        conn = sqlite3.connect(_DB_PATH)
        try:
            cur = conn.cursor()
            cur.execute(statement)
            cols = [d[0] for d in cur.description] if cur.description else []
            data = [dict(zip(cols, row)) for row in cur.fetchall()]
            return json.dumps({"success": True, "row_count": len(data), "data": data})
        except sqlite3.Error as exc:
            return json.dumps(
                {"success": False, "error": f"SQL error: {exc}", "row_count": 0, "data": []}
            )
        finally:
            conn.close()
