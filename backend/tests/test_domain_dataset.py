import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAINS_DIR = REPO_ROOT / "domains"


def _load_eval_module(name: str):
    path = REPO_ROOT / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_build_dataset = _load_eval_module("build_dataset")
_upload = _load_eval_module("upload_galileo_dataset")
DOMAINS = _build_dataset.discover_domains()


def test_domains_are_discovered() -> None:
    assert {"platform", "bank", "healthcare", "insurance", "restaurant"} <= set(DOMAINS)


def test_platform_system_prompt_matches_app() -> None:
    from app.agent.graph import SYSTEM_PROMPT

    prompt = json.loads(
        (DOMAINS_DIR / "platform" / "system_prompt.json").read_text()
    )["system_prompt"]
    assert prompt == SYSTEM_PROMPT


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_dataset_is_grounded(domain: str) -> None:
    domain_dir = DOMAINS_DIR / domain
    rows = _build_dataset.load_qa(domain_dir / "docs" / "qa.csv")
    assert rows
    assert _build_dataset.grounding_warnings(rows, domain_dir) == []


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_generated_dataset_matches_qa(domain: str) -> None:
    domain_dir = DOMAINS_DIR / domain
    qa_rows = _build_dataset.load_qa(domain_dir / "docs" / "qa.csv")

    csv_rows = _upload.read_dataset_csv(domain_dir / "dataset.csv")
    assert len(csv_rows) == len(qa_rows)
    assert csv_rows[0]["input"] == qa_rows[0]["question"]
    assert csv_rows[0]["output"] == qa_rows[0]["answer"]

    jsonl_lines = (domain_dir / "dataset.jsonl").read_text().splitlines()
    assert len(jsonl_lines) == len(qa_rows)
    first = json.loads(jsonl_lines[0])
    assert first["input"] == qa_rows[0]["question"]
    assert first["metadata"]["domain"] == domain
