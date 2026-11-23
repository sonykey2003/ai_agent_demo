import json
from pathlib import Path

import pytest

from app import domains as domains_mod
from app.agent import agent_control, tools
from app.agent.agent_control import ControlSteerError
from app.rag import ingest


def test_registry_includes_ecomm_and_reused_domains() -> None:
    domains = domains_mod.get_domains()
    assert {"ecomm", "bank", "healthcare", "insurance", "restaurant"} <= set(domains)
    assert domains_mod.DEFAULT_DOMAIN == "ecomm"


def test_ecomm_is_a_regular_domain_with_own_collection() -> None:
    ecomm = domains_mod.get_domains()["ecomm"]
    assert ecomm.label == "E-commerce"
    assert ecomm.collection == "ai_agent_demo_ecomm"
    assert ecomm.docs_dir.as_posix().endswith("domains/ecomm/docs")


def test_reused_domain_has_own_collection_and_prompt() -> None:
    domains = domains_mod.get_domains()
    bank = domains["bank"]
    assert bank.collection == "ai_agent_demo_bank"
    assert bank.docs_dir.as_posix().endswith("domains/bank/docs")
    assert bank.system_prompt
    assert bank.system_prompt != domains["ecomm"].system_prompt


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


def test_db_tools_are_scoped_to_their_domain() -> None:
    names = lambda d: {t.name for t in tools.tools_for_domain(d)}  # noqa: E731
    assert names("ecomm") == {"calculator", "query_order_db"}
    assert names("bank") == {"calculator", "query_customer_db"}
    assert names("healthcare") == {"calculator", "query_patient_db"}
    assert names("insurance") == {"calculator"}


@pytest.mark.parametrize(
    ("db_tool", "ledger"),
    [(tools.query_order_db, "orders"), (tools.query_patient_db, "prescriptions")],
)
def test_domain_db_reseeds_after_destructive_sql(db_tool, ledger) -> None:
    deleted = json.loads(db_tool.invoke({"sql": f"DELETE FROM {ledger}"}))
    assert deleted["rows_affected"] == 48

    count = json.loads(db_tool.invoke({"sql": f"SELECT count(*) n FROM {ledger}"}))
    assert count["data"] == [{"n": 48}]


def test_order_db_steer_returns_capped_retry(monkeypatch) -> None:
    def steer(sql: str) -> str:
        raise ControlSteerError(
            control_name="sql-limit", steering_context="Add a LIMIT of at most 5 rows."
        )

    monkeypatch.setattr(tools, "_query_orders_controlled", steer)
    result = json.loads(tools.query_order_db.invoke({"sql": "SELECT * FROM orders;"}))

    assert result["success"] is False
    assert result["rejected_sql"] == "SELECT * FROM orders;"
    assert result["retry_with"] == "SELECT * FROM orders LIMIT 5"


def test_native_logger_is_selected_per_domain(monkeypatch) -> None:
    bank_logger, ecomm_logger = object(), object()
    monkeypatch.setattr(agent_control, "_ENABLED", True)
    monkeypatch.setattr(agent_control, "_initialized", True)
    monkeypatch.setattr(
        agent_control, "_loggers", {"bank": bank_logger, "ecomm": ecomm_logger}
    )

    assert agent_control.get_ac_logger("ecomm") is ecomm_logger
    assert agent_control.get_ac_logger("bank") is bank_logger
    assert agent_control.get_ac_logger("healthcare") is None
