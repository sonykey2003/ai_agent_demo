"""Runtime registry of knowledge domains selectable from the UI.

Each domain maps to its own pgvector collection and (optionally) its own system
prompt. ``platform`` is the demo's own knowledge (``backend/knowledge`` -> the
default collection); the others are reused from the golden demo and ground on
their ``docs/qa.csv``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import get_settings

DEFAULT_DOMAIN = "platform"


@dataclass(frozen=True)
class Domain:
    name: str
    label: str
    collection: str
    docs_dir: Path
    system_prompt: str | None


def knowledge_dir() -> Path:
    """Return the platform knowledge directory (backend/knowledge)."""
    return Path(__file__).resolve().parents[1] / "knowledge"


def domains_root() -> Path:
    """Return the domains/ directory, resolved in dev and container layouts."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "domains"
        if candidate.is_dir():
            return candidate
    return here.parents[2] / "domains"


@lru_cache(maxsize=1)
def get_domains() -> dict[str, "Domain"]:
    settings = get_settings()
    root = domains_root()
    domains: dict[str, Domain] = {}
    if not root.is_dir():
        return domains
    for path in sorted(root.iterdir()):
        if not (path / "docs" / "qa.csv").is_file():
            continue
        name = path.name
        prompt_file = path / "system_prompt.json"
        system_prompt = None
        if prompt_file.is_file():
            system_prompt = json.loads(prompt_file.read_text(encoding="utf-8")).get(
                "system_prompt"
            )
        if name == DEFAULT_DOMAIN:
            collection = settings.rag_collection
            docs_dir = knowledge_dir()
        else:
            collection = f"ai_agent_demo_{name}"
            docs_dir = path / "docs"
        domains[name] = Domain(
            name=name,
            label=name.replace("-", " ").title(),
            collection=collection,
            docs_dir=docs_dir,
            system_prompt=system_prompt,
        )
    return domains
