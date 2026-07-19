"""Model adapters used by the LangChain RAG pipeline."""

from __future__ import annotations

import time
from functools import lru_cache
import re

from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import (
    DOCUMENT_PREFIX,
    EMBEDDING_MODEL,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
    GOOGLE_API_KEY,
    HF_TOKEN,
    MAX_RETRIES,
    QUERY_PREFIX,
    RETRY_DELAY_SECONDS,
)


def _require_hf_token() -> str:
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN in your .env first.")
    return HF_TOKEN


def _vector_to_list(vector: object) -> list[float]:
    if hasattr(vector, "reshape"):
        return vector.reshape(-1).tolist()
    return list(vector)  # type: ignore[arg-type]


_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def _is_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


class EmbeddingGemmaEmbeddings(Embeddings):
    """LangChain embeddings wrapper that preserves EmbeddingGemma prefixes."""

    def __init__(self) -> None:
        self._client = InferenceClient(
            provider="hf-inference",
            api_key=_require_hf_token(),
        )

    def _embed_with_retry(self, text: str, prefix: str) -> list[float]:
        payload = f"{prefix}{text}"
        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vector = self._client.feature_extraction(payload, model=EMBEDDING_MODEL)
                return _vector_to_list(vector)
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                print(f"  attempt {attempt} failed: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

        raise RuntimeError(f"Failed to embed text after {MAX_RETRIES} attempts: {last_error}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_with_retry(text, DOCUMENT_PREFIX) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if _is_arabic(text):
            return self._embed_with_retry(text, "")
        return self._embed_with_retry(text, QUERY_PREFIX)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return EmbeddingGemmaEmbeddings()


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    if not GOOGLE_API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env to use Gemini.")

    llm_kwargs = {
        "model": GEMINI_MODEL,
        "google_api_key": GOOGLE_API_KEY,
        "temperature": GEMINI_TEMPERATURE,
        "convert_system_message_to_human": False,
    }
    if GEMINI_MAX_OUTPUT_TOKENS:
        llm_kwargs["max_output_tokens"] = int(GEMINI_MAX_OUTPUT_TOKENS)

    return ChatGoogleGenerativeAI(**llm_kwargs)
