from pathlib import Path

from app import domains as domains_mod
from app.agent import tools
from app.config import get_settings
from app.rag import ingest


def test_registry_includes_platform_and_reused_domains() -> None:
    domains = domains_mod.get_domains()
    assert {"platform", "bank", "healthcare", "insurance", "restaurant"} <= set(domains)


def test_platform_uses_default_collection_and_knowledge_dir() -> None:
    platform = domains_mod.get_domains()["platform"]
    assert platform.collection == get_settings().rag_collection
    assert platform.docs_dir == domains_mod.knowledge_dir()


def test_reused_domain_has_own_collection_and_prompt() -> None:
    domains = domains_mod.get_domains()
    bank = domains["bank"]
    assert bank.collection == "ai_agent_demo_bank"
    assert bank.docs_dir.as_posix().endswith("domains/bank/docs")
    assert bank.system_prompt
    assert bank.system_prompt != domains["platform"].system_prompt


def test_ingest_loads_qa_csv_as_documents(tmp_path: Path) -> None:
    (tmp_path / "qa.csv").write_text(
        "question,answer\n\"What is X?\",\"X is a test.\"\n", encoding="utf-8"
    )
    docs = ingest._load_documents(tmp_path)

    assert len(docs) == 1
    assert "Question: What is X?" in docs[0].page_content
    assert "Answer: X is a test." in docs[0].page_content
    assert docs[0].metadata["title"] == "What is X?"


def test_rag_collection_scope_sets_and_restores() -> None:
    assert tools.active_rag_collection.get() is None
    with tools.rag_collection_scope("ai_agent_demo_bank"):
        assert tools.active_rag_collection.get() == "ai_agent_demo_bank"
    assert tools.active_rag_collection.get() is None
