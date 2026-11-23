"""
Basic unit tests for the LLMOps pipeline.
Add more comprehensive tests as needed.
"""
import pytest
from src.config import Config


def test_config_loaded():
    """Configuration exposes Azure settings without requiring secrets at import time."""
    assert hasattr(Config, "AZURE_OPENAI_API_KEY")
    assert Config.AZURE_OPENAI_API_VERSION


def test_vector_store_type():
    """Test vector store type configuration."""
    assert Config.VECTOR_STORE_TYPE == "azure_search"


def test_mlflow_config():
    """MLflow is optional while its experiment name always has a default."""
    assert Config.MLFLOW_EXPERIMENT_NAME is not None


def test_validate_reports_missing_required_settings(monkeypatch):
    for name in (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_DEPLOYMENT_NAME",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_KEY",
        "AZURE_SEARCH_INDEX_NAME",
    ):
        monkeypatch.setattr(Config, name, None)
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        Config.validate()
