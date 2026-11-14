"""Main entrypoint for the Secure Software Engineering RAG Assistant."""

import os
import uvicorn
from dotenv import load_dotenv

# Load environment configuration from .env
load_dotenv()

from app.api import app

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() in ("true", "1")

    print(f"===============================================================")
    print(f" Secure Software Engineering RAG Assistant")
    print(f" Running at: http://{host}:{port}")
    print(f"===============================================================")

    uvicorn.run("app.api:app", host=host, port=port, reload=debug)

