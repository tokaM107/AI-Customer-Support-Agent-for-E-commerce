"""Shared configuration for the RAG stack."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "noon_kb")
CHUNKS_PATH = PROJECT_ROOT / "chunks.json"

HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "google/embeddinggemma-300m")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_GEMINI_MODEL") or "gemini-3-flash-preview"
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))
GEMINI_MAX_OUTPUT_TOKENS = os.getenv("GEMINI_MAX_OUTPUT_TOKENS")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_SSL = os.getenv("CHROMA_SSL", "false").lower() == "true"

TOP_K = int(os.getenv("RAG_TOP_K", "5"))
MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "3"))
RETRY_DELAY_SECONDS = int(os.getenv("RAG_RETRY_DELAY_SECONDS", "5"))

DOCUMENT_PREFIX = "title: none | text: "
QUERY_PREFIX = "task: search result | query: "

DEFAULT_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "512"))
