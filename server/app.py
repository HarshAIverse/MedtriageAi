"""
server/app.py — OpenEnv multi-mode deployment entry point.

This module re-exports the FastAPI application from the project root so that
the `medical-triage-serve` CLI script and `uvicorn server.app:app` both work.

Usage:
    # via CLI script (after pip install -e .):
    medical-triage-serve

    # via uvicorn directly:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

    # via Docker CMD (in Dockerfile):
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 1
"""

from __future__ import annotations

# Re-export the FastAPI app from the root module.
# This keeps a single source of truth while satisfying the server/app.py
# path expected by the OpenEnv multi-mode deployment validator.
from app import app  # noqa: F401  (re-export)


def main() -> None:
    """Entrypoint for the `medical-triage-serve` CLI script."""
    import uvicorn

    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
