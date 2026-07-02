"""Generic offline evaluation harness (vendor-neutral).

Runs the same agent over a dataset and checks simple containment assertions.
This is intentionally framework-agnostic; swap in Ragas, promptfoo, or Galileo
Evaluate as a drop-in runner over the same datasets.

Usage:
    python eval/run_eval.py --provider local --dataset eval/datasets/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the backend package importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.agent.graph import build_agent  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run(dataset: Path, provider: str | None) -> bool:
    agent = build_agent(provider=provider)
    cases = load_cases(dataset)
    passed = 0
    for i, case in enumerate(cases, 1):
        result = agent.invoke({"messages": [("user", case["input"])]})
        answer = result["messages"][-1].content or ""
        expected = case.get("expected_contains", [])
        ok = all(s.lower() in answer.lower() for s in expected)
        passed += int(ok)
        print(f"[{i}] {'PASS' if ok else 'FAIL'} :: {case['input'][:60]}")
        if not ok:
            print(f"      expected to contain: {expected}")
            print(f"      got: {answer[:160]}")
    print(f"\n{passed}/{len(cases)} passed")
    return passed == len(cases)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline agent evaluation")
    parser.add_argument("--provider", default=None, help="openai | nim | local")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "datasets" / "smoke.jsonl"),
    )
    args = parser.parse_args()
    ok = run(Path(args.dataset), args.provider)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
