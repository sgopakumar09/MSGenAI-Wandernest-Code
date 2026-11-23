from langchain_core.documents import Document

from src.data_loader import TravelDataLoader
from src.ingestion import _document_id


def test_update_file_overrides_base_file_and_retains_source_metadata(tmp_path):
    data_dir = tmp_path / "data"
    updates_dir = tmp_path / "document_updates"
    data_dir.mkdir()
    updates_dir.mkdir()
    (data_dir / "travel.csv").write_text("policy\nold version\n", encoding="utf-8")
    (updates_dir / "travel.csv").write_text("policy\nupdated version\n", encoding="utf-8")
    loader = TravelDataLoader(data_dir=data_dir, updates_dir=updates_dir)

    documents = loader.load_all_travel_documents()
    chunks = loader.split_documents(documents)

    assert len(documents) == 1
    assert documents[0].page_content == "policy: updated version"
    assert documents[0].metadata["document_origin"] == "document_updates"
    assert chunks[0].metadata["chunk_index"] == 0


def test_reingested_source_chunk_keeps_its_index_id():
    metadata = {"source": "travel.pdf", "page": 2, "chunk_index": 1}
    old_chunk = Document(page_content="Old policy text", metadata=metadata)
    revised_chunk = Document(page_content="Revised policy text", metadata=metadata)

    assert _document_id(old_chunk) == _document_id(revised_chunk)


def test_filename_categories_prioritize_refund_and_privacy_topics():
    assert (
        TravelDataLoader._categorize_document(
            "U.S. Department of Transportation - Refunds and Other Consumer Protections.pdf"
        )
        == "refund_policies"
    )
    assert (
        TravelDataLoader._categorize_document(
            "U.S. Department of Transportation - Air Consumer Privacy.pdf"
        )
        == "privacy_policies"
    )
    assert (
        TravelDataLoader._categorize_document(
            "air-india-general-booking-policies-oct2025.pdf"
        )
        == "booking_policies"
    )
