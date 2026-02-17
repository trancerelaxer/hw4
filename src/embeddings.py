import os
import hashlib
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
EMBEDDING_MODEL = os.getenv("LM_STUDIO_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5")
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache"
CACHE_FILE = CACHE_DIR / "embedding_cache.json"
Embedding = list[float]
_cache: dict[str, Embedding] | None = None

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
)


def _to_embedding(value: Any) -> Embedding | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(x, (int, float)) for x in value):
        return None
    return [float(x) for x in value]


def _load_cache() -> dict[str, Embedding]:
    global _cache
    if _cache is not None:
        return _cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
            if isinstance(data, dict):
                parsed_cache: dict[str, Embedding] = {}
                for key, value in data.items():
                    if not isinstance(key, str):
                        continue
                    embedding = _to_embedding(value)
                    if embedding is not None:
                        parsed_cache[key] = embedding
                _cache = parsed_cache
            else:
                _cache = {}
        except Exception:
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache(cache: dict[str, Embedding]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=True))


def _cache_key(text: str) -> str:
    raw = f"{EMBEDDING_MODEL}\n{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def embed_texts(texts: list[str]) -> list[Embedding]:
    if not texts:
        return []

    cache = _load_cache()
    embeddings: list[Embedding | None] = [None] * len(texts)
    missing_texts = []
    missing_indices = []

    for idx, text in enumerate(texts):
        key = _cache_key(text)
        cached_embedding = cache.get(key)
        if cached_embedding is not None:
            embeddings[idx] = cached_embedding
        else:
            missing_indices.append(idx)
            missing_texts.append(text)

    cache_updated = False
    try:
        if missing_texts:
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=missing_texts)
            for idx, item in zip(missing_indices, response.data):
                embedding = item.embedding
                embeddings[idx] = embedding
                cache[_cache_key(texts[idx])] = embedding
                cache_updated = True
    except Exception as exc:
        raise RuntimeError(
            "Failed to create embeddings via LM Studio. "
            "Ensure LM Studio is running, the local server is enabled, and an embedding model is loaded."
        ) from exc

    if cache_updated:
        _save_cache(cache)

    if any(embedding is None for embedding in embeddings):
        raise RuntimeError("Failed to build embeddings for one or more inputs.")

    return [embedding for embedding in embeddings if embedding is not None]


def embed_query(query: str) -> Embedding:
    return embed_texts([query])[0]
