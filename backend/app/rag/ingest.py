"""Load local knowledge documents into PostgreSQL/pgvector."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings
from ..domains import DEFAULT_DOMAIN, get_domains, knowledge_dir
from .vector_store import get_embeddings

SUPPORTED_SUFFIXES = {".md", ".txt", ".csv"}


def default_source_dir() -> Path:
    """Return the bundled knowledge directory in local and container layouts."""
    return knowledge_dir()


def _documents_from_file(path: Path, source_dir: Path) -> list[Document]:
    rel = path.relative_to(source_dir).as_posix()
    if path.suffix.lower() == ".csv":
        docs: list[Document] = []
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                question = (row.get("question") or "").strip()
                answer = (row.get("answer") or "").strip()
                if question and answer:
                    docs.append(
                        Document(
                            page_content=f"Question: {question}\nAnswer: {answer}",
                            metadata={"source": rel, "title": question},
                        )
                    )
        return docs
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    title = next(
        (
            line.lstrip("#").strip()
            for line in content.splitlines()
            if line.startswith("#")
        ),
        path.stem.replace("-", " ").title(),
    )
    return [Document(page_content=content, metadata={"source": rel, "title": title})]


def _load_documents(source_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            documents.extend(_documents_from_file(path, source_dir))
    return documents


def _chunk_id(document: Document) -> str:
    identity = f"{document.metadata['source']}\0{document.page_content}"
    return hashlib.sha256(identity.encode()).hexdigest()


def ingest(source_dir: Path, *, reset: bool = False, collection: str | None = None) -> int:
    settings = get_settings()
    documents = _load_documents(source_dir)
    if not documents:
        raise ValueError(f"No Markdown, text, or qa.csv documents found in {source_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    chunk_ids = [_chunk_id(chunk) for chunk in chunks]
    for chunk, chunk_id in zip(chunks, chunk_ids, strict=True):
        chunk.metadata["chunk_id"] = chunk_id
        source_stem = Path(chunk.metadata["source"]).stem
        chunk.metadata["citation_id"] = (
            f"{source_stem}-{chunk.metadata.get('start_index', 0)}"
        )
    store = PGVector(
        embeddings=get_embeddings(),
        collection_name=collection or settings.rag_collection,
        connection=settings.postgres_url,
        use_jsonb=True,
        pre_delete_collection=reset,
    )
    store.add_documents(chunks, ids=chunk_ids)
    return len(chunks)


def ingest_domain(name: str, *, reset: bool = True) -> int:
    """Index one domain's knowledge into its own pgvector collection."""
    domain = get_domains()[name]
    return ingest(domain.docs_dir, reset=reset, collection=domain.collection)


def ingest_all(*, reset: bool = True) -> dict[str, int]:
    """Index every registered domain; returns chunk counts per domain."""
    return {name: ingest_domain(name, reset=reset) for name in get_domains()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        nargs="?",
        type=Path,
        default=default_source_dir(),
        help="Directory containing Markdown or text files",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and rebuild the configured collection before ingestion",
    )
    parser.add_argument(
        "--domain",
        help="Ingest a registered domain into its own collection (overrides source_dir).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest every registered domain into its own collection.",
    )
    args = parser.parse_args()
    if args.all:
        counts = ingest_all(reset=args.reset or True)
        for name, count in counts.items():
            print(f"Indexed {count} chunks into {get_domains()[name].collection}")
        return
    if args.domain:
        count = ingest_domain(args.domain, reset=args.reset or True)
        print(f"Indexed {count} chunks into {get_domains()[args.domain].collection}")
        return
    count = ingest(args.source_dir, reset=args.reset)
    print(f"Indexed {count} chunks into {get_settings().rag_collection}")


if __name__ == "__main__":
    main()