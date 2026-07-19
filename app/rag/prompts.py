"""Prompt templates for the Noon customer support assistant."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """You are Noon’s customer support assistant for e-commerce help.
Use only the supplied context to answer the user’s question.

Rules:
- Be concise, accurate, and professional.
- If the context does not contain the answer, say that you could not find it in the knowledge base.
- Do not invent policies, prices, timelines, or procedures.
- When useful, mention the source article and section numbers from the context.
"""


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_prompt}\n\nRetrieved context:\n{context}\n\nAnswer with a helpful customer support response.",
        ),
        ("human", "Question:\n{question}"),
    ]
).partial(system_prompt=SYSTEM_PROMPT)


def format_document(document: Document) -> str:
    metadata = document.metadata or {}
    source = metadata.get("source", "unknown source")
    part = metadata.get("part")
    sub_part = metadata.get("sub_part")

    reference_bits = [f"source={source}"]
    if part is not None:
        reference_bits.append(f"part={part}")
    if sub_part is not None:
        reference_bits.append(f"sub_part={sub_part}")

    return f"[{'; '.join(reference_bits)}]\n{document.page_content.strip()}"


def format_context(documents: list[Document]) -> str:
    if not documents:
        return "No context was retrieved."
    return "\n\n---\n\n".join(format_document(document) for document in documents)
