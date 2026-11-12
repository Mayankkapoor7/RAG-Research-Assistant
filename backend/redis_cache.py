"""
backend/redis_cache.py
──────────────────────
Semantic query cache backed by Redis.

Workflow
────────
1. On query arrival, embed the query with text-embedding-3-small.
2. Scan all existing cache entries and compute cosine similarity.
3. Return the best-matching (answer, score) pair if any entry exists.
4. The caller decides whether the score is a HIT (≥ 0.9) or MISS (< 0.85).

Storage layout (Redis hash per entry)
──────────────────────────────────────
  Key   : redis_sem_cache:<uuid4>
  Fields:
    embedding  – raw 32-bit float bytes (4 bytes × 1536 dims = 6144 bytes)
    answer     – UTF-8 encoded answer string
    query      – UTF-8 encoded original query (for debugging / inspection)

All entries carry a TTL of REDIS_CACHE_TTL_SECONDS (default 86400 = 24 h).

If Redis is unreachable, every operation degrades gracefully (returns None /
no-ops), so the RAG pipeline continues without caching.
"""

import os
import struct
import uuid
import logging
from typing import Optional, Tuple

import numpy as np
import redis
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379")
REDIS_CACHE_TTL_SECONDS: int = int(os.environ.get("REDIS_CACHE_TTL_SECONDS", "86400"))
REDIS_HIT_THRESHOLD: float = float(os.environ.get("REDIS_HIT_THRESHOLD", "0.9"))
REDIS_MISS_THRESHOLD: float = float(os.environ.get("REDIS_MISS_THRESHOLD", "0.85"))

CACHE_KEY_PREFIX = "redis_sem_cache:"
EMBEDDING_DIM = 1536  # text-embedding-3-small

# ── Singletons ────────────────────────────────────────────────────────────────

# Reuse the same embedding model as vector_store.py to avoid creating a
# second OpenAI client with different settings.
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def _get_client() -> redis.Redis:
    """Return a Redis client connected to REDIS_URL."""
    return redis.from_url(REDIS_URL, decode_responses=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_bytes(embedding: list[float]) -> bytes:
    """Serialize a float list to compact binary (little-endian 32-bit floats)."""
    return struct.pack(f"<{len(embedding)}f", *embedding)


def _from_bytes(data: bytes) -> list[float]:
    """Deserialize binary back to a float list."""
    n = len(data) // 4
    return list(struct.unpack(f"<{n}f", data))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors. Returns 0.0 on zero-norm."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


# ── Public API ────────────────────────────────────────────────────────────────

def get(query: str) -> Optional[Tuple[str, float]]:
    """
    Look up a query in the semantic cache.

    Returns
    -------
    (answer, similarity_score)  if at least one cached entry exists
    None                        if the cache is empty or Redis is unavailable

    The caller must apply the hit / miss thresholds:
        score >= REDIS_HIT_THRESHOLD   → serve cached answer
        score <  REDIS_MISS_THRESHOLD  → run full RAG pipeline
        in-between                     → also run pipeline (freshness wins)
    """
    try:
        client = _get_client()
        query_embedding = _embeddings.embed_query(query)

        keys = list(client.scan_iter(f"{CACHE_KEY_PREFIX}*"))
        if not keys:
            return None

        best_score = -1.0
        best_answer: Optional[str] = None

        for key in keys:
            entry = client.hgetall(key)
            if not entry:
                continue
            raw_emb: Optional[bytes] = entry.get(b"embedding")
            raw_ans: Optional[bytes] = entry.get(b"answer")
            if raw_emb is None or raw_ans is None:
                continue

            cached_embedding = _from_bytes(raw_emb)
            score = _cosine_similarity(query_embedding, cached_embedding)

            if score > best_score:
                best_score = score
                best_answer = raw_ans.decode("utf-8")

        if best_answer is not None:
            logger.info("[redis_cache] best similarity=%.4f for query=%r", best_score, query[:80])
            return best_answer, best_score

        return None

    except Exception as exc:
        logger.warning("[redis_cache] get() failed (Redis unavailable?): %s", exc)
        return None



def store(query: str, answer: str) -> None:
    """
    Persist a query-answer pair to Redis with a TTL.

    Silently no-ops if Redis is unavailable.
    """
    try:
        client = _get_client()
        embedding = _embeddings.embed_query(query)
        emb_bytes = _to_bytes(embedding)

        key = f"{CACHE_KEY_PREFIX}{uuid.uuid4()}"
        client.hset(
            key,
            mapping={
                "embedding": emb_bytes,
                "answer": answer.encode("utf-8"),
                "query": query.encode("utf-8"),
            },
        )
        client.expire(key, REDIS_CACHE_TTL_SECONDS)
        logger.info("[redis_cache] stored key=%s (TTL=%ds)", key, REDIS_CACHE_TTL_SECONDS)

    except Exception as exc:
        logger.warning("[redis_cache] store() failed (Redis unavailable?): %s", exc)
