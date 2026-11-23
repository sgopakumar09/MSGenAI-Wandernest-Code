"""Load, categorize, and chunk the travel knowledge base."""
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPDATE_DIR = PROJECT_ROOT / "document_updates"


class TravelDataLoader:
    """Loads PDFs and CSVs from the base data and document update folders."""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        updates_dir: Optional[Path] = None,
    ):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.updates_dir = Path(updates_dir) if updates_dir else UPDATE_DIR
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    def _directories(self) -> List[Path]:
        return [path for path in (self.data_dir, self.updates_dir) if path.is_dir()]

    def _document_files(self, extension: str) -> List[Path]:
        latest_by_name = {}
        for directory in self._directories():
            for path in directory.glob(f"*{extension}"):
                latest_by_name[path.name.casefold()] = path
        return sorted(latest_by_name.values(), key=lambda path: path.name.casefold())

    def _source_metadata(self, path: Path) -> dict:
        origin = "document_updates" if self.updates_dir in path.parents else "data"
        return {
            "source": path.name,
            "file_type": path.suffix.lstrip(".").lower(),
            "category": self._categorize_document(path.name),
            "document_origin": origin,
        }

    def load_pdfs_from_data_directory(self) -> List[Document]:
        """Load every PDF in the base and updates directories."""
        documents = []
        pdf_files = self._document_files(".pdf")
        logger.debug("Found %d PDF files across data directories", len(pdf_files))
        failures = []

        for index, pdf_file in enumerate(pdf_files, start=1):
            try:
                logger.debug("Loading PDF %d/%d: %s", index, len(pdf_files), pdf_file.name)
                docs = PyPDFLoader(str(pdf_file)).load()
                metadata = self._source_metadata(pdf_file)
                for doc in docs:
                    doc.metadata.update(metadata)
                documents.extend(docs)
            except Exception as exc:
                failures.append(f"{pdf_file}: {exc}")

        if failures:
            raise RuntimeError("Failed to load PDF documents:\n" + "\n".join(failures))
        return documents

    @staticmethod
    def _categorize_document(filename: str) -> str:
        filename_lower = filename.lower()
        if "refund" in filename_lower:
            return "refund_policies"
        if "privacy" in filename_lower:
            return "privacy_policies"
        if "booking" in filename_lower:
            return "booking_policies"
        if "air-india" in filename_lower or "ai-schedule" in filename_lower:
            return "air_india_policies"
        if "u.s. department" in filename_lower or "transportation" in filename_lower:
            return "us_dot_regulations"
        if "policy" in filename_lower:
            return "booking_policies"
        return "general"

    def load_csvs_from_data_directory(self) -> List[Document]:
        """Load CSV rows from the base and updates directories."""
        documents = []
        csv_files = self._document_files(".csv")
        failures = []
        for index, csv_file in enumerate(csv_files, start=1):
            try:
                logger.debug("Loading CSV %d/%d: %s", index, len(csv_files), csv_file.name)
                dataframe = pd.read_csv(csv_file)
                metadata = self._source_metadata(csv_file)
                for index, row in dataframe.iterrows():
                    content = " | ".join(f"{column}: {value}" for column, value in row.items())
                    documents.append(
                        Document(
                            page_content=content,
                            metadata={**metadata, "row_index": int(index)},
                        )
                    )
            except Exception as exc:
                failures.append(f"{csv_file}: {exc}")
        if failures:
            raise RuntimeError("Failed to load CSV documents:\n" + "\n".join(failures))
        return documents

    def load_all_travel_documents(self) -> List[Document]:
        documents = self.load_pdfs_from_data_directory()
        documents.extend(self.load_csvs_from_data_directory())
        logger.debug("Loaded %d travel documents", len(documents))
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split loaded documents while retaining their source metadata."""
        chunks = self.text_splitter.split_documents(documents)
        chunk_counts = {}
        for chunk in chunks:
            identity = (
                chunk.metadata.get("source"),
                chunk.metadata.get("page", chunk.metadata.get("row_index", 0)),
            )
            chunk.metadata["chunk_index"] = chunk_counts.get(identity, 0)
            chunk_counts[identity] = chunk.metadata["chunk_index"] + 1
        return chunks


if __name__ == "__main__":
    loader = TravelDataLoader()
    documents = loader.load_all_travel_documents()
    chunks = loader.split_documents(documents)
    print(f"Loaded {len(documents)} documents into {len(chunks)} chunks")
