"""Application entrypoint for the Noon customer support assistant."""

from __future__ import annotations
import sys
import os
from app.rag.chain import run_cli


def main() -> None:
	run_cli()


if __name__ == "__main__":
	main()
