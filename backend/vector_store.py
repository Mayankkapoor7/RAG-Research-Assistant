import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1536  # text-embedding-3-small
LOCAL_QDRANT_PATH = "./qdrant_db"

# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.environ.get("OPENAI_API_KEY"),
)

cache_dir = Path("./embedding_cache")
cache_dir.mkdir(exist_ok=True)
embedding_file_store = LocalFileStore(str(cache_dir))

embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

_qdrant_client_instance = None


def get_qdrant_client() -> QdrantClient:
    """Return a working Qdrant client, preferring remote if reachable, falling back to local persistent storage."""
    global _qdrant_client_instance
    if _qdrant_client_instance is not None:
        return _qdrant_client_instance

    remote_url = os.environ.get("QDRANT_URL", "").strip()
    remote_key = os.environ.get("QDRANT_API_KEY", "").strip()

    if remote_url and not remote_url.startswith("local"):
        try:
            client = QdrantClient(url=remote_url, api_key=remote_key, timeout=4)
            # Test ping
            client.get_collections()
            _qdrant_client_instance = client
            return client
        except Exception as e:
            logger.warning("Remote Qdrant unavailable (%s). Falling back to local storage at '%s'.", e, LOCAL_QDRANT_PATH)

    os.makedirs(LOCAL_QDRANT_PATH, exist_ok=True)
    _qdrant_client_instance = QdrantClient(path=LOCAL_QDRANT_PATH)
    return _qdrant_client_instance


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    cleaned = session_id.replace("-", "_")
    return f"papeer_{cleaned}"


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    client = get_qdrant_client()
    collection_name = get_collection_name(session_id)
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    if not docs:
        return
    vs = get_vectorstore(session_id)
    vs.add_documents(docs)


def list_papers(session_id: str) -> list[str]:
    """Return unique document titles loaded in this session."""
    client = get_qdrant_client()
    collection_name = get_collection_name(session_id)
    if not client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            meta = (point.payload or {}).get("metadata", {})
            title = meta.get("file_name") or meta.get("title") or meta.get("source")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


SIMILARITY_THRESHOLD = 0.30


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    """Retrieve top-k relevant document chunks for the given session."""
    try:
        vs = get_vectorstore(session_id)
        results = vs.similarity_search_with_score(query, k=k)
        if not results:
            return []
        filtered = [doc for doc, score in results if score >= SIMILARITY_THRESHOLD]
        if not filtered and results:
            # If no document met the similarity threshold, return the best candidate
            return [results[0][0]]
        return filtered
    except Exception as e:
        logger.error("Error during vector store search: %s", e)
        return []
