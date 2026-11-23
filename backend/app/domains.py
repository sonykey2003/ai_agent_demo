"""Runtime registry of knowledge domains selectable from the UI.

Each domain maps to its own pgvector collection (``ai_agent_demo_<domain>``) and
(optionally) its own system prompt, and grounds on its ``docs/`` (``qa.csv``).
``ecomm`` (online home furnishings store) is the default; ``bank``,
``healthcare``, ``insurance``, and ``restaurant`` are reused from the golden demo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_DOMAIN = "ecomm"

# Display labels that title-casing the directory name would get wrong.
_LABELS = {"ecomm": "E-commerce"}


@dataclass(frozen=True)
class Domain:
    name: str
    label: str
    collection: str
    docs_dir: Path
    system_prompt: str | None


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
        domains[name] = Domain(
            name=name,
            label=_LABELS.get(name, name.replace("-", " ").title()),
            collection=f"ai_agent_demo_{name}",
            docs_dir=path / "docs",
            system_prompt=system_prompt,
        )
    return domains
