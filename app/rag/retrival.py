"""Backward-compatible retrieval entrypoint.

Keep this file so existing commands still work, but route all behavior
through the LangChain-based orchestration in app.rag.chain.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.rag.chain import run_cli


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
