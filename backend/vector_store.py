import os

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings # Caching the embeddings, so we don't have to pay for the same embeddings again
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 1536  # text-embedding-3-small

# ── Singletons ────────────────────────────────────────────────────────────────

base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embedding_file_store = LocalFileStore("./embedding_cache/") # Creates a folder on hard drive where we will save our data
embeddings = CacheBackedEmbeddings.from_bytes_store( # Save them as raw binary bytes
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model, #  By setting the namespace to the model's name, it neatly categorizes the cache files 
    query_embedding_cache=True, # Caching the embeddings
    key_encoder="blake2b", # Hashing the embeddings
)

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)


# ── Collection ───────────────────────────────────────────────────────────────

def get_collection_name(session_id: str) -> str:
    return f"papeer_{session_id.replace('-', '_')}" # Replace hyphens with underscores as streamlit generates in hyphens where as qdrant prefers underscores


def get_vectorstore(session_id: str) -> QdrantVectorStore:
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
    )


# ── Public API ───────────────────────────────────────────────────────────────

def add_paper(docs: list[Document], session_id: str) -> None:
    get_vectorstore(session_id).add_documents(docs)


def list_papers(session_id: str) -> list[str]: 
    # Used in loaded documents functionality, to tell which documents are loaded in the current session
    collection_name = get_collection_name(session_id)
    if not qdrant_client.collection_exists(collection_name):
        return []
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=collection_name,
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(query: str, session_id: str, k: int = 4) -> list[Document]:
    return get_vectorstore(session_id).similarity_search(query, k=k)
