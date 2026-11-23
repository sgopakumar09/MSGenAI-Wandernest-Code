"""Environment-backed application configuration."""
import os
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


def _integer_setting(name: str, default: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be zero or greater")
    return parsed


class Config:
    """Configuration loaded from environment variables (including a local .env)."""

    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
    AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME", "travel-kb-index")
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "azure_search").lower()

    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "travel-documents")

    AZURE_CONTENT_SAFETY_ENDPOINT = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT")
    AZURE_CONTENT_SAFETY_KEY = os.getenv("AZURE_CONTENT_SAFETY_KEY")
    APPLICATIONINSIGHTS_CONNECTION_STRING = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "wanderlust-travel-chatbot")
    INGESTION_LIMIT = _integer_setting("INGESTION_LIMIT", 0)

    @classmethod
    def validate(cls, *, require_openai: bool = True, require_search: bool = True) -> None:
        """Raise an actionable error listing missing service settings."""
        required = []
        if require_openai:
            required.extend(
                [
                    "AZURE_OPENAI_API_KEY",
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_OPENAI_DEPLOYMENT_NAME",
                    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
                ]
            )
        if require_search:
            required.extend(["AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_SEARCH_INDEX_NAME"])
        missing = [name for name in required if not getattr(cls, name)]
        if cls.VECTOR_STORE_TYPE != "azure_search":
            raise ValueError("VECTOR_STORE_TYPE must be 'azure_search'.")
        if missing:
            raise ValueError("Missing required configuration: " + ", ".join(missing))
        for name in ("AZURE_OPENAI_ENDPOINT", "AZURE_SEARCH_ENDPOINT"):
            endpoint = getattr(cls, name)
            if endpoint and (
                urlparse(endpoint).scheme not in {"https", "http"}
                or not urlparse(endpoint).netloc
            ):
                raise ValueError(f"{name} must be a valid service URL.")

