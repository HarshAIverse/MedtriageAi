"""
server/app.py — OpenEnv multi-mode deployment entry point.
Re-exports the FastAPI app and provides the main() CLI entrypoint.
"""

from app import app  # noqa: F401  (re-export for uvicorn server.app:app)


def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
    )


if __name__ == "__main__":
    main()
