"""Azure AI Search vector store initialization."""
from langchain_community.vectorstores import AzureSearch

from src.config import Config


def get_vector_store(embedding_function):
    """Create the configured Azure AI Search LangChain vector store."""
    Config.validate(require_openai=False, require_search=True)
    if embedding_function is None:
        raise ValueError("An Azure OpenAI embedding function is required.")
    return AzureSearch(
        azure_search_endpoint=Config.AZURE_SEARCH_ENDPOINT,
        azure_search_key=Config.AZURE_SEARCH_KEY,
        index_name=Config.AZURE_SEARCH_INDEX_NAME,
        embedding_function=embedding_function.embed_query,
    )
