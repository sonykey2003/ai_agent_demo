"""Tiny SQLite order database backing the ecomm-domain query tool (demo only).

Mirrors ``bank_db``: re-seeded in the container's ephemeral storage before every
query, so a destructive statement only affects that one call. Three tables:

- ``customers`` has name/email/phone, so it drives the output-PII control demo.
- ``orders`` is a deliberately PII-free order ledger (48 rows), so the SQL
  row-limit steer can be demoed without an output-PII control withholding the
  answer.
- ``products`` is the catalogue (article number, price, stock, warehouse), for
  ordinary stock and price lookups.

Fabricated data only; product names are generic descriptions, not a brand's.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .demo_db import run_on_fresh_db

_DB_PATH = Path("/tmp/ecomm_orders.db")

# (id, name, email, phone, member_tier, city) — fabricated demo data only.
_CUSTOMERS = [
    ("C-2001", "Maja Lindqvist", "maja.lindqvist@example.com", "+1-415-555-0101", "Home Club", "San Francisco"),
    ("C-2002", "Oliver Grant", "oliver.grant@example.com", "+1-206-555-0102", "Guest", "Seattle"),
    ("C-2003", "Priya Nair", "priya.nair@example.com", "+1-312-555-0103", "Home Club", "Chicago"),
    ("C-2004", "Lucas Moreau", "lucas.moreau@example.com", "+1-617-555-0104", "Business", "Boston"),
    ("C-2005", "Hannah Kim", "hannah.kim@example.com", "+1-512-555-0105", "Home Club", "Austin"),
    ("C-2006", "Diego Alvarez", "diego.alvarez@example.com", "+1-303-555-0106", "Guest", "Denver"),
    ("C-2007", "Sofia Rossi", "sofia.rossi@example.com", "+1-646-555-0107", "Business", "New York"),
    ("C-2008", "Noah Bennett", "noah.bennett@example.com", "+1-503-555-0108", "Home Club", "Portland"),
]

# (sku, name, category, price, stock_qty, warehouse, assembly_required)
_PRODUCTS = [
    ("204.118.32", "Oak-veneer 5-shelf bookcase", "Storage", 89.00, 142, "Central DC", 1),
    ("305.229.41", "3-seat fabric sofa, light grey", "Living room", 649.00, 18, "North DC", 1),
    ("103.447.56", "Queen pocket-spring mattress, medium-firm", "Bedroom", 399.00, 0, "South DC", 0),
    ("402.336.17", "6-drawer dresser, white stain", "Bedroom", 279.00, 37, "Central DC", 1),
    ("501.552.80", "Extendable dining table, birch, 4-6 seats", "Dining", 349.00, 9, "North DC", 1),
    ("602.613.24", "Stackable dining chair, solid pine", "Dining", 45.00, 260, "Central DC", 1),
    ("703.774.65", "Height-adjustable desk, 140x70 cm", "Workspace", 429.00, 3, "South DC", 1),
    ("804.885.09", "Ergonomic office chair, mesh back", "Workspace", 199.00, 54, "North DC", 1),
    ("905.996.38", "Kitchen base cabinet, 60 cm, matte white", "Kitchen", 135.00, 210, "Central DC", 1),
    ("106.107.72", "Induction hob, 4 zones", "Kitchen", 499.00, 12, "South DC", 0),
    ("207.218.53", "LED floor lamp, dimmable", "Lighting", 59.99, 88, "Central DC", 0),
    ("308.329.14", "Blackout curtains, 1 pair, dark blue", "Textiles", 39.99, 175, "North DC", 0),
    ("409.430.95", "Wardrobe with sliding doors, 200 cm", "Storage", 579.00, 0, "Central DC", 1),
    ("510.541.66", "Bunk bed frame, solid pine", "Children", 299.00, 21, "South DC", 1),
    ("611.652.27", "Outdoor lounge chair, acacia", "Outdoor", 129.00, 46, "North DC", 1),
    ("712.763.88", "Wall-anchoring kit for tall furniture", "Spare parts", 0.00, 999, "Central DC", 0),
]

_STATUSES = ["Delivered", "Shipped", "Processing", "Ready for pickup", "Delivered", "Returned"]
_FULFILMENT = ["Home delivery", "Click & collect", "Parcel"]


def _seed_orders() -> list[tuple[str, str, str, str, int, float, str, str]]:
    """48 deterministic order lines (6 per customer) — enough that a LIMIT matters."""
    rows = []
    per_customer = 6
    for c_idx, customer in enumerate(_CUSTOMERS):
        for n in range(per_customer):
            i = c_idx * per_customer + n
            sku, _name, _cat, price, *_ = _PRODUCTS[(c_idx * 3 + n * 5) % len(_PRODUCTS)]
            qty = 1 + (i % 4 if price < 100 else 0)
            rows.append(
                (
                    f"O-{10401 + i}",
                    customer[0],
                    f"2026-10-{1 + (c_idx * 6 + n * 4) % 28:02d}",
                    sku,
                    qty,
                    round(price * qty, 2),
                    _STATUSES[(c_idx + n) % len(_STATUSES)],
                    _FULFILMENT[(c_idx + 2 * n) % len(_FULFILMENT)],
                )
            )
    return rows


def _seed_db(conn: sqlite3.Connection) -> None:
    """Drop and recreate the three tables with the fabricated demo rows."""
    conn.executescript(
        """
        DROP TABLE IF EXISTS customers;
        CREATE TABLE customers (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            member_tier TEXT NOT NULL,
            city TEXT NOT NULL
        );
        DROP TABLE IF EXISTS products;
        CREATE TABLE products (
            sku TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_qty INTEGER NOT NULL,
            warehouse TEXT NOT NULL,
            assembly_required INTEGER NOT NULL
        );
        DROP TABLE IF EXISTS orders;
        CREATE TABLE orders (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            order_date TEXT NOT NULL,
            sku TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            fulfilment TEXT NOT NULL
        );
        """
    )
    conn.executemany(
        "INSERT INTO customers (id, name, email, phone, member_tier, city) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        _CUSTOMERS,
    )
    conn.executemany(
        "INSERT INTO products "
        "(sku, name, category, price, stock_qty, warehouse, assembly_required) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        _PRODUCTS,
    )
    conn.executemany(
        "INSERT INTO orders "
        "(id, customer_id, order_date, sku, quantity, amount, status, fulfilment) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        _seed_orders(),
    )


def query_orders(sql: str) -> str:
    """Execute the given SQL against the demo store tables, whatever it is.

    No statement allow-list: Galileo Agent Control is the only thing that may
    refuse a query. Returns ``{"success", "row_count", "data": [...]}`` as JSON.
    """
    return run_on_fresh_db(_DB_PATH, _seed_db, sql)
