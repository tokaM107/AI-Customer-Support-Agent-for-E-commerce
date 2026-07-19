"""Embed chunks.json and upsert them into the Noon Chroma collection.

This script uses the shared LangChain embedding adapter from app.rag.llm so
the ingestion and query paths stay aligned on the same EmbeddingGemma
prefixes and retry behavior.
"""

from __future__ import annotations

if __package__ in {None, ""}:
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os

import chromadb

from app.rag.config import CHUNKS_PATH, COLLECTION_NAME
from app.rag.llm import get_embeddings

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

UPSERT_BATCH_SIZE = 50  # chunks per Chroma upsert call


# ---------------------------------------------------------------------------
# Chroma client -- Cloud or self-hosted, based on what's in .env
# ---------------------------------------------------------------------------

def get_chroma_client():
    if os.environ.get("CHROMA_API_KEY"):
        return chromadb.CloudClient(
            api_key=os.environ["CHROMA_API_KEY"],
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"],
        )
    if os.environ.get("CHROMA_HOST"):
        return chromadb.HttpClient(
            host=os.environ["CHROMA_HOST"],
            port=int(os.environ.get("CHROMA_PORT", 8000)),
            ssl=os.environ.get("CHROMA_SSL", "false").lower() == "true",
        )
    raise RuntimeError(
        "No Chroma connection details found in .env. Set either "
        "CHROMA_API_KEY/CHROMA_TENANT/CHROMA_DATABASE (Chroma Cloud) or "
        "CHROMA_HOST (self-hosted)."
    )


# ---------------------------------------------------------------------------
def make_chunk_id(chunk, index):
    meta = chunk["metadata"]
    parts = [str(meta.get("source", "unknown")), str(meta.get("part", index))]
    if "sub_part" in meta:
        parts.append(f"sub{meta['sub_part']}")
    return "::".join(parts)


def clean_metadata(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool -- drop anything else."""
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    embeddings = get_embeddings()
    chroma_client = get_chroma_client()
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Using Chroma collection '{COLLECTION_NAME}'")

    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    embedded_count = 0
    skipped_count = 0

    def flush_batch():
        if not batch_ids:
            return
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas,
        )
        print(f"  upserted batch of {len(batch_ids)}")
        batch_ids.clear()
        batch_embeddings.clear()
        batch_documents.clear()
        batch_metadatas.clear()

    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        meta = chunk["metadata"]
        print(f"[{i}/{len(chunks)}] embedding {meta.get('source', '?')} "
              f"part {meta.get('part', '?')}")
        try:
            vector = embeddings.embed_documents([text])[0]
        except RuntimeError as e:
            print(f"  SKIPPING this chunk: {e}")
            skipped_count += 1
            continue

        batch_ids.append(make_chunk_id(chunk, i))
        batch_embeddings.append(vector)
        batch_documents.append(text)
        batch_metadatas.append(clean_metadata(meta))
        embedded_count += 1

        if len(batch_ids) >= UPSERT_BATCH_SIZE:
            flush_batch()

    flush_batch()

    print(f"\nDone. Embedded {embedded_count} chunks, skipped {skipped_count}.")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} items.")


if __name__ == "__main__":
    main()