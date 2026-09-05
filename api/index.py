from fastapi import FastAPI

from backend.main import (
    home,
    status,
    upload_documents,
    query_document,
    reset,
)


app = FastAPI(
    title="RAG Document Chat API",
    description="Multi-document RAG backend using FAISS and Groq",
    version="1.0.0",
)


app.add_api_route(
    "/",
    home,
    methods=["GET"],
)

app.add_api_route(
    "/status",
    status,
    methods=["GET"],
)

app.add_api_route(
    "/upload",
    upload_documents,
    methods=["POST"],
)

app.add_api_route(
    "/query",
    query_document,
    methods=["POST"],
)

app.add_api_route(
    "/reset",
    reset,
    methods=["POST"],
)