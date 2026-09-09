"""Read-only dump of the Agent Control controls bound to a log stream.

The SDK has no control CRUD; the server does (see $AGENT_CONTROL_URL/openapi.json).
Usage: .venv/bin/python eval/inspect_controls.py [log_stream_id]
"""

from __future__ import annotations

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE = os.environ.get("AGENT_CONTROL_URL", "").rstrip("/")
KEY = os.environ.get("GALILEO_API_KEY", "")
STREAM = sys.argv[1] if len(sys.argv) > 1 else "e4285877-2fb2-4f8e-b48e-91a87b8065f1"
HEADERS = {"Galileo-API-Key": KEY, "Accept": "application/json"}

_CLIENT = httpx.Client(headers=HEADERS, timeout=30)


def get(url: str):
    resp = _CLIENT.get(url)
    if resp.status_code != 200:
        return {"_error": resp.status_code, "_body": resp.text[:400]}
    return resp.json()


def main() -> None:
    if not BASE or not KEY:
        sys.exit("AGENT_CONTROL_URL / GALILEO_API_KEY missing from .env")
    listing = get(
        f"{BASE}/api/v1/control-bindings"
        f"?target_type=log_stream&target_id={STREAM}&limit=100"
    )
    print("=== bindings ===")
    print(json.dumps(listing, indent=2)[:4000])

    items = listing if isinstance(listing, list) else (
        listing.get("bindings") or listing.get("control_bindings") or listing.get("data") or []
    )
    for item in items if isinstance(items, list) else []:
        cid = item.get("control_id") or item.get("id")
        if cid is None:
            continue
        print(f"\n=== control {cid} ===")
        print(json.dumps(get(f"{BASE}/api/v1/controls/{cid}"), indent=2)[:6000])


if __name__ == "__main__":
    main()
