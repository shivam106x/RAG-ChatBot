from pathlib import Path
import os

import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document as DocxDocument

from sentence_transformers import SentenceTransformer
from openai import OpenAI


# ============================================================
# Environment
# ============================================================

load_dotenv()


# ============================================================
# Configuration
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5
COMPARISON_CHUNKS_PER_DOCUMENT = 6


# ============================================================
# Local Embedding Model
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)


# ============================================================
# Document Loading
# ============================================================

def load_document(file_path):
    """
    Load PDF, DOCX, or TXT.

    Returns dictionaries containing:
        text
        source
        page
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    documents = []

    # -------------------------
    # PDF
    # -------------------------

    if extension == ".pdf":

        reader = PdfReader(
            str(file_path)
        )

        for page_number, page in enumerate(
            reader.pages
        ):

            text = page.extract_text() or ""

            text = text.strip()

            if text:

                documents.append(
                    {
                        "text": text,
                        "source": str(file_path),
                        "page": page_number
                    }
                )

    # -------------------------
    # DOCX
    # -------------------------

    elif extension == ".docx":

        document = DocxDocument(
            str(file_path)
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                paragraphs.append(text)

        full_text = "\n".join(
            paragraphs
        )

        if full_text:

            documents.append(
                {
                    "text": full_text,
                    "source": str(file_path),
                    "page": None
                }
            )

    # -------------------------
    # TXT
    # -------------------------

    elif extension == ".txt":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read().strip()

        if text:

            documents.append(
                {
                    "text": text,
                    "source": str(file_path),
                    "page": None
                }
            )

    else:

        raise ValueError(
            "Unsupported file type. "
            "Use PDF, DOCX, or TXT."
        )

    if not documents:

        raise ValueError(
            f"No readable text found in {file_path.name}"
        )

    return documents


# ============================================================
# Chunking
# ============================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
):
    """
    Split text into overlapping chunks.
    """

    text = text.strip()

    if not text:

        return []

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(
                chunk
            )

        if end >= len(text):

            break

        start = end - chunk_overlap

    return chunks


def split_documents(documents):
    """
    Create chunks while preserving source metadata.
    """

    chunks = []

    for document in documents:

        text_chunks = split_text(
            document["text"]
        )

        for text_chunk in text_chunks:

            chunks.append(
                {
                    "text": text_chunk,
                    "source": document["source"],
                    "page": document["page"]
                }
            )

    if not chunks:

        raise ValueError(
            "No chunks were created."
        )

    return chunks


# ============================================================
# Embeddings
# ============================================================

def embed_texts(texts):
    """
    Generate normalized local embeddings.
    """

    vectors = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return np.asarray(
        vectors,
        dtype="float32"
    )


# ============================================================
# FAISS Vector Store
# ============================================================

def create_vector_store(chunks):
    """
    Create FAISS index and keep vectors locally
    so comparison retrieval can be source-aware.
    """

    if not chunks:

        raise ValueError(
            "Cannot create vector store without chunks."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectors = embed_texts(
        texts
    )

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        vectors
    )

    return {
        "index": index,
        "chunks": chunks,
        "vectors": vectors
    }


# ============================================================
# Detect comparison questions
# ============================================================

def is_comparison_question(question):
    """
    Detect whether the user is asking for
    comparison/shared information across documents.
    """

    comparison_terms = {
        "common",
        "both",
        "compare",
        "comparison",
        "difference",
        "differences",
        "similar",
        "similarities",
        "same",
        "shared",
        "overlap",
        "versus",
        "vs",
        "between"
    }

    words = set(
        question.lower()
        .replace("?", " ")
        .replace(",", " ")
        .replace(".", " ")
        .split()
    )

    return bool(
        words.intersection(
            comparison_terms
        )
    )


# ============================================================
# Normal Retrieval
# ============================================================

def normal_search(
    vector_store,
    question,
    k=TOP_K
):
    """
    Standard global top-k retrieval.
    """

    query_vector = embed_texts(
        [question]
    )

    total_chunks = (
        vector_store["index"].ntotal
    )

    if total_chunks == 0:

        return []

    k = min(
        k,
        total_chunks
    )

    scores, indices = (
        vector_store["index"].search(
            query_vector,
            k
        )
    )

    results = []

    for score, position in zip(
        scores[0],
        indices[0]
    ):

        if position == -1:

            continue

        chunk = vector_store[
            "chunks"
        ][position]

        results.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# Document-Aware Comparison Retrieval
# ============================================================

def comparison_search(
    vector_store,
    question
):
    """
    For comparison questions, retrieve the best
    chunks from EACH source document.

    This prevents a single document from dominating
    the retrieved context.
    """

    query_vector = embed_texts(
        [question]
    )[0]

    chunks = vector_store[
        "chunks"
    ]

    vectors = vector_store[
        "vectors"
    ]

    # Cosine similarity because vectors are normalized.
    scores = vectors @ query_vector

    # Group chunk positions by source.
    source_positions = {}

    for position, chunk in enumerate(chunks):

        source = chunk["source"]

        if source not in source_positions:

            source_positions[source] = []

        source_positions[source].append(
            position
        )

    results = []

    # Retrieve best chunks from every document.
    for source, positions in (
        source_positions.items()
    ):

        ranked_positions = sorted(
            positions,
            key=lambda position: scores[position],
            reverse=True
        )

        selected_positions = ranked_positions[
            :COMPARISON_CHUNKS_PER_DOCUMENT
        ]

        for position in selected_positions:

            chunk = chunks[position]

            results.append(
                {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "score": float(
                        scores[position]
                    )
                }
            )

    return results


# ============================================================
# Main Retrieval Function
# ============================================================

def search_documents(
    vector_store,
    question,
    k=TOP_K,
    force_comparison=False
):
    """
    Use standard retrieval for normal questions
    and document-aware retrieval for comparison questions.
    """

    if not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    comparison_requested = (
        force_comparison or
        is_comparison_question(question)
    )

    if comparison_requested:

        return comparison_search(
            vector_store,
            question
        )

    return normal_search(
        vector_store,
        question,
        k
    )


# ============================================================
# Source Handling
# ============================================================

def build_sources(results):
    """
    Deduplicate source/page combinations.
    """

    sources = []

    seen = set()

    for result in results:

        source = result["source"]
        page = result["page"]

        key = (
            source,
            page
        )

        if key in seen:

            continue

        seen.add(
            key
        )

        sources.append(
            {
                "source": source,
                "page": page
            }
        )

    return sources


# ============================================================
# Groq Generation
# ============================================================

def generate_answer(
    question,
    results
):
    """
    Generate answer using only retrieved context.
    """

    if not results:

        return (
            "I don't know based on the provided documents."
        )

    comparison = is_comparison_question(
        question
    )

    unique_document_sources = {
        result["source"]
        for result in results
    }

    # --------------------------------------------------------
    # Safety check for comparisons
    # --------------------------------------------------------

    if comparison and len(
        unique_document_sources
    ) < 2:

        return (
            "I don't have enough evidence from "
            "multiple uploaded documents to make "
            "this comparison."
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for result in results:

        filename = Path(
            result["source"]
        ).name

        page = result["page"]

        if page is not None:

            source_label = (
                f"{filename}, Page {page + 1}"
            )

        else:

            source_label = filename

        context_parts.append(
            f"[SOURCE: {source_label}]\n"
            f"{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document question-answering assistant.

Use ONLY the information contained in the
provided document context.

Rules:
1. Never use outside knowledge.
2. Never invent facts.
3. Every factual claim must be supported by
   the supplied context.
4. If the answer cannot be supported, say:
   "I don't know based on the provided documents."
5. For comparison questions, explicitly compare
   information from the different source documents.
6. Do not claim something is common to both documents
   unless the context provides evidence from both.
7. Keep the answer concise and factual.

DOCUMENT CONTEXT:

{context}

QUESTION:

{question}

ANSWER:
"""

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY not found in .env"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer strictly from the "
                    "provided document context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        reasoning_effort="low",
        max_tokens=500
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:

        return (
            "I could not generate an answer "
            "from the provided documents."
        )

    return answer.strip()


# ============================================================
# Full RAG Query
# ============================================================

def answer_question(
    vector_store,
    question,
    force_comparison=False
):
    """
    Complete RAG workflow.
    """

    results = search_documents(
        vector_store,
        question,
        force_comparison=force_comparison
    )

    answer = generate_answer(
        question,
        results
    )

    sources = build_sources(
        results
    )

    return (
        answer,
        sources,
        results
    )


# ============================================================
# Multi-document ingestion
# ============================================================

def build_vector_store(file_paths):
    """
    Load all documents, chunk them,
    then create one FAISS index.
    """

    all_documents = []
    all_chunks = []

    for file_path in file_paths:

        documents = load_document(
            file_path
        )

        chunks = split_documents(
            documents
        )

        all_documents.extend(
            documents
        )

        all_chunks.extend(
            chunks
        )

    if not all_chunks:

        raise ValueError(
            "No content found in uploaded documents."
        )

    vector_store = create_vector_store(
        all_chunks
    )

    return (
        vector_store,
        all_chunks,
        all_documents
    )


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":

    print(
        "Starting local RAG pipeline..."
    )

    vector_store, chunks, documents = (
        build_vector_store(
            ["test.txt"]
        )
    )

    print(
        f"Documents loaded: {len(documents)}"
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print(
        "Vector store created!"
    )

    question = (
        "What is FAISS used for?"
    )

    print(
        f"\nQuestion: {question}"
    )

    answer, sources, results = (
        answer_question(
            vector_store,
            question
        )
    )

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")

    for source in sources:

        filename = Path(
            source["source"]
        ).name

        page = source["page"]

        if page is not None:

            print(
                f"- {filename} | "
                f"Page {page + 1}"
            )

        else:

            print(
                f"- {filename}"
            )