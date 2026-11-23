import logging
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src import ingestion


def test_ingestion_logs_stages_and_success_summary(monkeypatch, caplog):
    document = Document(page_content="Travel policy")
    loader = MagicMock()
    loader.load_all_travel_documents.return_value = [document]
    loader.split_documents.return_value = [document]
    engine = MagicMock()
    engine.search_by_text.return_value = ([document], "Air India travel policy")
    monkeypatch.setattr(ingestion.Config, "validate", lambda: None)
    monkeypatch.setattr(ingestion.Config, "MLFLOW_TRACKING_URI", None)
    monkeypatch.setattr(ingestion.Config, "INGESTION_LIMIT", 0)
    monkeypatch.setattr(ingestion, "TravelDataLoader", lambda: loader)
    monkeypatch.setattr(ingestion, "TravelSearchEngine", lambda: engine)

    with caplog.at_level(logging.INFO, logger="src.ingestion"):
        result = ingestion.ingest_travel_documents()

    assert result == 0
    messages = caplog.messages
    assert "Starting travel document ingestion..." in messages
    assert "Loading PDF and CSV documents..." in messages
    assert "Prepared 1 chunks for indexing." in messages
    assert "INGESTION RESULTS" in messages
    assert "INGESTION PASSED" in messages


def test_console_logging_suppresses_sdk_http_noise(monkeypatch):
    logger_names = (
        "azure.core.pipeline.policies.http_logging_policy",
        "httpx",
        "httpcore",
    )
    sdk_loggers = [logging.getLogger(name) for name in logger_names]
    original_levels = [logger.level for logger in sdk_loggers]
    for logger, level in zip(sdk_loggers, original_levels):
        monkeypatch.setattr(logger, "level", level)
    monkeypatch.setattr(ingestion.logging, "basicConfig", lambda **kwargs: None)

    ingestion._configure_console_logging()

    assert all(logger.level >= logging.WARNING for logger in sdk_loggers)
