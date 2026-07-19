"""High-level orchestration for retrieval and answer generation."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from .config import (
    CHROMA_API_KEY,
    CHROMA_DATABASE,
    CHROMA_HOST,
    CHROMA_PORT,
    CHROMA_SSL,
    CHROMA_TENANT,
    COLLECTION_NAME,
    TOP_K,
)
from .llm import get_embeddings, get_llm
from .prompts import RAG_PROMPT, format_context


def _build_vectorstore() -> Chroma:
    embedding_function = get_embeddings()

    if CHROMA_API_KEY:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            chroma_cloud_api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )

    if CHROMA_HOST:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embedding_function,
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=CHROMA_SSL,
        )

    raise RuntimeError(
        "No Chroma connection details found in .env. Set either "
        "CHROMA_API_KEY/CHROMA_TENANT/CHROMA_DATABASE (Chroma Cloud) or "
        "CHROMA_HOST (self-hosted)."
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return _build_vectorstore()


def get_retriever(top_k: int = TOP_K):
    return get_vectorstore().as_retriever(search_kwargs={"k": top_k})


@lru_cache(maxsize=8)
def build_answer_runnable(top_k: int = TOP_K) -> RunnableLambda:
    retriever = get_retriever(top_k)
    llm_chain = RAG_PROMPT | get_llm() | StrOutputParser()

    def _answer_question(question: str) -> dict:
        question = str(question).strip()
        documents = retriever.invoke(question)
        answer = llm_chain.invoke(
            {
                "question": question,
                "context": format_context(documents),
            }
        ).strip()
        return {
            "question": question,
            "answer": answer,
            "documents": documents,
        }

    return RunnableLambda(_answer_question)


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    return build_answer_runnable(top_k=top_k).invoke(question)


def print_answer(result: dict) -> None:
    print("\n=== Answer ===")
    print(result["answer"])

    documents = result.get("documents", [])
    if not documents:
        return

    print("\n=== Sources ===")
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata or {}
        source = metadata.get("source", "unknown source")
        part = metadata.get("part")
        sub_part = metadata.get("sub_part")
        labels = [source]
        if part is not None:
            labels.append(f"part {part}")
        if sub_part is not None:
            labels.append(f"sub {sub_part}")
        print(f"{index}. {' | '.join(labels)}")


def run_cli(argv: list[str] | None = None) -> None:
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)

    if arguments:
        query = " ".join(arguments).strip()
        if query:
            print_answer(answer_question(query))
        return

    print("Type a question (or 'quit' to exit):")
    while True:
        try:
            query = input("\n> ").strip()
        except EOFError:
            print()
            break

        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        print_answer(answer_question(query))
