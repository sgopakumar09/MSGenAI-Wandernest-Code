"""Batch-load base and updated travel documents into Azure AI Search."""
import logging
import uuid

import mlflow
from tqdm import tqdm

from src.config import Config
from src.data_loader import TravelDataLoader
from src.search_engine import TravelSearchEngine

logger = logging.getLogger(__name__)


def _configure_console_logging() -> None:
    """Show ingestion progress while suppressing per-request SDK logs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    for logger_name in (
        "azure.core.pipeline.policies.http_logging_policy",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def _document_id(document) -> str:
    """Make re-ingestion of a source page/chunk an upsert rather than a duplicate."""
    identity = (
        f"{document.metadata.get('source', '')}:"
        f"{document.metadata.get('page', document.metadata.get('row_index', 0))}:"
        f"{document.metadata.get('chunk_index', 0)}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, identity))


def ingest_travel_documents() -> int:
    """Load PDFs and CSVs from data/ and document_updates/, then verify indexing."""
    engine = None
    mlflow_active = False
    documents = []
    chunks = []
    indexed_count = 0
    try:
        logger.info("Starting travel document ingestion...")
        logger.info("Validating Azure configuration...")
        Config.validate()
        loader = TravelDataLoader()
        logger.info("Initializing Azure AI Search...")
        engine = TravelSearchEngine()
        if Config.MLFLOW_TRACKING_URI:
            try:
                mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
                mlflow.start_run(run_name="document_ingestion")
                mlflow_active = True
            except Exception as exc:
                logger.warning("MLflow tracking unavailable; continuing without tracking: %s", exc)

        logger.info("Loading PDF and CSV documents...")
        documents = loader.load_all_travel_documents()
        if not documents:
            raise RuntimeError("No PDFs or CSV files found in data/ or document_updates/.")
        logger.info("Loaded %d source documents.", len(documents))

        logger.info("Splitting documents into searchable chunks...")
        chunks = loader.split_documents(documents)
        if Config.INGESTION_LIMIT:
            chunks = chunks[: Config.INGESTION_LIMIT]
        if not chunks:
            raise RuntimeError("No document chunks were produced for indexing.")
        logger.info("Prepared %d chunks for indexing.", len(chunks))

        batch_size = 50
        failures = []
        logger.info("Indexing documents in batches of up to %d...", batch_size)
        for start in tqdm(range(0, len(chunks), batch_size), desc="Indexing"):
            batch = chunks[start : start + batch_size]
            ids = [_document_id(document) for document in batch]
            try:
                engine.vector_store.add_documents(documents=batch, ids=ids)
                indexed_count += len(batch)
            except Exception as exc:
                failures.append(f"Batch {start // batch_size + 1}: {exc}")

        if mlflow_active:
            mlflow.log_param("loaded_documents", len(documents))
            mlflow.log_param("chunk_size", loader.text_splitter._chunk_size)
            mlflow.log_param("chunk_overlap", loader.text_splitter._chunk_overlap)
            mlflow.log_metric("ingested_count", indexed_count)
            mlflow.log_metric("failed_count", len(chunks) - indexed_count)

        if failures:
            raise RuntimeError(
                f"Indexing failed for {len(chunks) - indexed_count} chunks:\n"
                + "\n".join(failures)
            )

        logger.info("Verifying indexed documents with a sample search...")
        results, _ = engine.search_by_text("Air India travel policy", k=3)
        if not results:
            raise RuntimeError("Index verification failed: the test query returned no results.")
        logger.info("=" * 60)
        logger.info("INGESTION RESULTS")
        logger.info("=" * 60)
        logger.info("Source documents loaded: %d", len(documents))
        logger.info("Chunks indexed: %d", indexed_count)
        logger.info("Verification search results: %d", len(results))
        logger.info("INGESTION PASSED")
        return 0
    except Exception:
        logger.error(
            "INGESTION FAILED (loaded %d documents; indexed %d of %d chunks)",
            len(documents),
            indexed_count,
            len(chunks),
        )
        logger.exception("Document ingestion failed")
        return 1
    finally:
        if mlflow_active:
            mlflow.end_run()


if __name__ == "__main__":
    _configure_console_logging()
    raise SystemExit(ingest_travel_documents())
