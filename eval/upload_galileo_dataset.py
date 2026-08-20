"""Upload a domain dataset to Galileo and (optionally) run a baseline experiment.

Prepares the artifacts from ``eval/build_dataset.py`` for evaluation and
context-adherence / coherence baselining. This is a Galileo-specific admin
helper — the vendor-neutral app never imports the Galileo SDK.

Usage:
    pip install -r eval/requirements-galileo.txt

    # .env must contain GALILEO_API_KEY and GALILEO_PROJECT (and, for --baseline,
    # a reachable model provider). GALILEO_CONSOLE_URL is derived from
    # GALILEO_OTEL_ENDPOINT when unset.
    python eval/upload_galileo_dataset.py --domain platform             # upload dataset
    python eval/upload_galileo_dataset.py --domain platform --preview   # show, don't upload
    python eval/upload_galileo_dataset.py --domain platform --baseline --provider local
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv() -> None:
    """Load REPO_ROOT/.env into os.environ without overriding existing values."""
    env_path = REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _resolve_console_url() -> None:
    if os.environ.get("GALILEO_CONSOLE_URL"):
        return
    host = urlparse(os.environ.get("GALILEO_OTEL_ENDPOINT", "")).netloc
    if host.startswith("api."):
        os.environ["GALILEO_CONSOLE_URL"] = f"https://{host.replace('api.', 'console.', 1)}"


def read_dataset_csv(dataset_file: Path) -> list[dict]:
    rows: list[dict] = []
    with dataset_file.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("input") and row.get("output"):
                rows.append({"input": row["input"].strip(), "output": row["output"].strip()})
    if not rows:
        raise ValueError(f"No input/output rows found in {dataset_file}")
    return rows


def _dataset_name(domain: str) -> str:
    return f"{domain.title()} Domain Dataset"


def _baseline_metrics() -> list:
    """Return the available RAG/adherence scorers for a baseline experiment."""
    from galileo_core.schemas.shared.scorers.scorer_name import (  # noqa: PLC0415
        ScorerName as GalileoScorers,
    )

    wanted = [
        "context_adherence",
        "ground_truth_adherence",
        "chunk_attribution_utilization",
        "completeness",
    ]
    return [getattr(GalileoScorers, name) for name in wanted if hasattr(GalileoScorers, name)]


def _make_experiment_function(provider: str | None):
    """Build the app agent once and return a callable Galileo can score."""
    sys.path.insert(0, str(REPO_ROOT / "backend"))
    from app.agent.graph import build_agent  # noqa: PLC0415

    agent = build_agent(provider=provider)

    def run_case(row) -> str:
        user_input = row["input"] if isinstance(row, dict) else str(row)
        result = agent.invoke(
            {"messages": [("user", user_input)]},
            config={"configurable": {"thread_id": "galileo-baseline"}},
        )
        return result["messages"][-1].content or ""

    return run_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="platform")
    parser.add_argument("--preview", action="store_true", help="Show rows without uploading.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="After upload, run a baseline experiment scoring adherence metrics.",
    )
    parser.add_argument("--provider", default="local", help="openai | nim | local | local_gemma")
    args = parser.parse_args()

    _load_dotenv()
    _resolve_console_url()

    dataset_file = REPO_ROOT / "domains" / args.domain / "dataset.csv"
    if not dataset_file.is_file():
        print(f"ERROR: {dataset_file} not found. Run: python eval/build_dataset.py --domain {args.domain}")
        return 1

    content = read_dataset_csv(dataset_file)
    project = os.environ.get("GALILEO_PROJECT", "default")
    dataset_name = _dataset_name(args.domain)

    print(f"Domain  : {args.domain}")
    print(f"Project : {project}")
    print(f"Dataset : {dataset_name} ({len(content)} rows)")
    if args.preview:
        for i, row in enumerate(content[:3], 1):
            print(f"  [{i}] input : {row['input']}")
            print(f"      output: {row['output'][:80]}...")
        print("\nPreview only; nothing uploaded.")
        return 0

    if not os.environ.get("GALILEO_API_KEY"):
        print("ERROR: GALILEO_API_KEY is not set (see .env).")
        return 1

    try:
        from galileo.datasets import create_dataset, get_dataset  # noqa: PLC0415
    except ImportError:
        print("ERROR: Galileo SDK missing. Run: pip install -r eval/requirements-galileo.txt")
        return 1

    try:
        dataset = create_dataset(name=dataset_name, content=content, project_name=project)
        print(f"Created dataset '{dataset_name}' (id={getattr(dataset, 'id', '?')}).")
    except Exception:  # noqa: BLE001
        dataset = get_dataset(name=dataset_name, project_name=project)
        print(f"Dataset '{dataset_name}' already exists; reusing (id={getattr(dataset, 'id', '?')}).")

    if not args.baseline:
        print("\nDone. Open the dataset in Galileo, or re-run with --baseline to score it.")
        return 0

    try:
        from galileo.experiments import run_experiment  # noqa: PLC0415
    except ImportError:
        print("ERROR: Galileo SDK missing the experiments module.")
        return 1

    metrics = _baseline_metrics()
    if not metrics:
        print("ERROR: no adherence scorers available in this Galileo SDK version.")
        return 1

    print(f"Running baseline experiment with provider '{args.provider}' and metrics: "
          f"{[getattr(m, 'value', str(m)) for m in metrics]}")
    results = run_experiment(
        f"{args.domain}-baseline",
        dataset=dataset,
        function=_make_experiment_function(args.provider),
        metrics=metrics,
        project=project,
    )
    print(f"Baseline experiment submitted: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
