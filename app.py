"""Launcher for the Noon FastAPI RAG interface."""

from __future__ import annotations

from app.api import app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)