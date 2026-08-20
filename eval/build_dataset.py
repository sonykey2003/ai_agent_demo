"""Build Galileo-ready evaluation datasets from a domain's Q&A file.

Reads ``domains/<domain>/docs/qa.csv`` (question, answer, source) and writes two
upload artifacts next to it:

  - ``domains/<domain>/dataset.csv``   -> ``input,output`` for the Galileo UI upload
  - ``domains/<domain>/dataset.jsonl`` -> ``{input, output, metadata}`` for the SDK

Each answer is checked for grounding against the knowledge file named in its
``source`` column, so the reference outputs stay faithful for context-adherence
and coherence baselining.

Usage:
    python eval/build_dataset.py                      # domain defaults to "platform"
    python eval/build_dataset.py --domain platform    # regenerate after editing qa.csv
    python eval/build_dataset.py --check              # fail if any answer looks ungrounded
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = REPO_ROOT / "backend" / "knowledge"

# Common words carry no grounding signal; drop them before measuring overlap.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "of", "in", "on",
    "for", "and", "or", "with", "as", "at", "by", "from", "this", "that", "it",
    "its", "which", "what", "when", "how", "do", "does", "did", "can", "will",
    "your", "you", "their", "they", "using", "used", "use", "any", "not", "no",
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def load_qa(qa_path: Path) -> list[dict]:
    rows: list[dict] = []
    with qa_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader, 1):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            source = (row.get("source") or "").strip()
            if not question or not answer:
                raise ValueError(f"{qa_path} row {i}: question and answer are required")
            rows.append({"question": question, "answer": answer, "source": source})
    if not rows:
        raise ValueError(f"No rows found in {qa_path}")
    return rows


def _corpus_tokens(docs_dir: Path) -> set[str]:
    """All tokens across a domain's own docs (used when rows have no source)."""
    tokens: set[str] = set()
    for path in docs_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".csv"}:
            tokens |= _tokens(path.read_text(encoding="utf-8"))
    return tokens


def grounding_warnings(rows: list[dict], domain_dir: Path) -> list[str]:
    """Return a warning per answer that is weakly supported by its knowledge.

    Rows with a ``source`` column are grounded against that named file under
    ``backend/knowledge`` (the platform domain). Rows without a source are
    self-grounded against the domain's own ``docs/`` corpus (the reused golden
    domains, whose knowledge is the qa.csv itself).
    """
    warnings: list[str] = []
    has_source = any(row["source"] for row in rows)
    corpus = None if has_source else _corpus_tokens(domain_dir / "docs")
    for i, row in enumerate(rows, 1):
        answer_tokens = _tokens(row["answer"]) - _STOPWORDS
        if corpus is not None:
            overlap = len(answer_tokens & corpus) / max(len(answer_tokens), 1)
            if overlap < 0.5:
                warnings.append(f"row {i}: only {overlap:.0%} of answer terms appear in docs/")
            continue
        source = row["source"]
        path = KNOWLEDGE_DIR / source
        if not path.is_file():
            warnings.append(f"row {i}: source '{source}' not found under {KNOWLEDGE_DIR}")
            continue
        doc_tokens = _tokens(path.read_text(encoding="utf-8"))
        overlap = len(answer_tokens & doc_tokens) / max(len(answer_tokens), 1)
        if overlap < 0.5:
            warnings.append(
                f"row {i}: only {overlap:.0%} of answer terms appear in {source}"
            )
    return warnings


def write_csv(rows: list[dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL)
        writer.writerow(["input", "output"])
        for row in rows:
            writer.writerow([row["question"], row["answer"]])


def write_jsonl(rows: list[dict], domain: str, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {
                "input": row["question"],
                "output": row["answer"],
                "metadata": {"domain": domain, "source": row["source"] or "qa.csv"},
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(domain: str, *, check: bool) -> int:
    domain_dir = REPO_ROOT / "domains" / domain
    qa_path = domain_dir / "docs" / "qa.csv"
    if not qa_path.is_file():
        raise FileNotFoundError(f"Q&A file not found: {qa_path}")

    rows = load_qa(qa_path)
    warnings = grounding_warnings(rows, domain_dir)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if check and warnings:
        print(f"\n{len(warnings)} grounding warning(s); failing due to --check.")
        return 1

    write_csv(rows, domain_dir / "dataset.csv")
    write_jsonl(rows, domain, domain_dir / "dataset.jsonl")
    print(
        f"Wrote {len(rows)} rows to "
        f"{domain_dir / 'dataset.csv'} and {domain_dir / 'dataset.jsonl'}"
    )
    return 0


def discover_domains() -> list[str]:
    """Return every domain that has a docs/qa.csv, sorted by name."""
    domains_dir = REPO_ROOT / "domains"
    return sorted(
        p.name for p in domains_dir.iterdir()
        if (p / "docs" / "qa.csv").is_file()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="platform")
    parser.add_argument("--all", action="store_true", help="Build every domain.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if any answer looks ungrounded in its source.",
    )
    args = parser.parse_args()
    domains = discover_domains() if args.all else [args.domain]
    exit_code = 0
    for domain in domains:
        print(f"== {domain} ==")
        exit_code |= build(domain, check=args.check)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
