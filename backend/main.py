from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from rag_pipeline import (
    load_document,
    split_documents,
    create_vector_store,
    answer_question
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="RAG Document Chat API",
    description="Multi-document RAG backend using FAISS and Groq",
    version="1.0.0"
)


# ============================================================
# In-memory knowledge base
# ============================================================

vector_store = None
all_chunks = []
uploaded_documents = []


# ============================================================
# Request model
# ============================================================

class QueryRequest(BaseModel):
    question: str


# ============================================================
# Health check
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RAG Document Chat API is running.",
        "documents_loaded": len(uploaded_documents),
        "total_chunks": len(all_chunks)
    }


# ============================================================
# Knowledge base status
# ============================================================

@app.get("/status")
def status():

    return {
        "ready": vector_store is not None,
        "documents": uploaded_documents,
        "document_count": len(uploaded_documents),
        "chunk_count": len(all_chunks)
    }


# ============================================================
# Upload multiple documents
# ============================================================

@app.post("/upload")
async def upload_documents(
    files: list[UploadFile] = File(...)
):

    global vector_store
    global all_chunks
    global uploaded_documents

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".txt"
    }

    if not files:

        raise HTTPException(
            status_code=400,
            detail="Please upload at least one document."
        )

    upload_folder = Path("uploads")
    upload_folder.mkdir(
        exist_ok=True
    )

    new_chunks = []
    new_document_names = []

    try:

        # ----------------------------------------------------
        # Process all uploaded files
        # ----------------------------------------------------

        for file in files:

            if not file.filename:

                raise HTTPException(
                    status_code=400,
                    detail="A file is missing its filename."
                )

            filename = Path(
                file.filename
            ).name

            extension = Path(
                filename
            ).suffix.lower()

            if extension not in allowed_extensions:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unsupported file type: {filename}. "
                        "Use PDF, DOCX, or TXT."
                    )
                )

            file_path = (
                upload_folder / filename
            )

            # ------------------------------------------------
            # Save uploaded file
            # ------------------------------------------------

            contents = await file.read()

            with open(
                file_path,
                "wb"
            ) as output_file:

                output_file.write(
                    contents
                )

            # ------------------------------------------------
            # Extract document
            # ------------------------------------------------

            documents = load_document(
                file_path
            )

            if not documents:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No readable text found in "
                        f"{filename}."
                    )
                )

            # ------------------------------------------------
            # Create chunks
            # ------------------------------------------------

            chunks = split_documents(
                documents
            )

            if not chunks:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"No chunks created for "
                        f"{filename}."
                    )
                )

            new_chunks.extend(
                chunks
            )

            new_document_names.append(
                filename
            )

        # ----------------------------------------------------
        # Add chunks to knowledge base
        # ----------------------------------------------------

        all_chunks.extend(
            new_chunks
        )

        # ----------------------------------------------------
        # Build FAISS index
        # ----------------------------------------------------

        vector_store = create_vector_store(
            all_chunks
        )

        # ----------------------------------------------------
        # Track documents
        # ----------------------------------------------------

        for filename in new_document_names:

            if filename not in uploaded_documents:

                uploaded_documents.append(
                    filename
                )

        return {
            "message": "Documents indexed successfully.",
            "documents_uploaded": new_document_names,
            "chunks_created": len(new_chunks),
            "total_documents": len(uploaded_documents),
            "total_chunks": len(all_chunks)
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error processing documents: {str(e)}"
            )
        )


# ============================================================
# Query
# ============================================================

@app.post("/query")
def query_document(
    request: QueryRequest
):

    global vector_store

    # --------------------------------------------------------
    # Check knowledge base
    # --------------------------------------------------------

    if vector_store is None:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload at least one "
                "document first."
            )
        )

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        # ----------------------------------------------------
        # Run RAG pipeline
        # ----------------------------------------------------

        answer, sources, results = (
            answer_question(
                vector_store,
                request.question
            )
        )

        # ----------------------------------------------------
        # Return response
        # ----------------------------------------------------

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(results)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Error answering question: {str(e)}"
            )
        )


# ============================================================
# Reset knowledge base
# ============================================================

@app.post("/reset")
def reset():

    global vector_store
    global all_chunks
    global uploaded_documents

    vector_store = None
    all_chunks = []
    uploaded_documents = []

    return {
        "message": (
            "Knowledge base reset successfully."
        )
    }