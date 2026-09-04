import requests
import streamlit as st


# ==================================================
# Configuration
# ==================================================

import os
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8001"
)

# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="RAG Document Chat",
    page_icon="📚",
    layout="centered"
)


# ==================================================
# Header
# ==================================================

st.title("📚 RAG Document Chat")

st.caption(
    "Multi-document Retrieval-Augmented Generation "
    "using FAISS and Groq."
)


# ==================================================
# Helper functions
# ==================================================

def get_status():

    try:

        response = requests.get(
            f"{API_URL}/status",
            timeout=10
        )

        if response.status_code == 200:

            return response.json()

    except requests.exceptions.RequestException:

        pass

    return None


# ==================================================
# Knowledge base status
# ==================================================

status_data = get_status()

if status_data and status_data["ready"]:

    st.success(
        f"🟢 Knowledge base ready | "
        f"{status_data['document_count']} document(s) | "
        f"{status_data['chunk_count']} chunks"
    )

elif status_data:

    st.info(
        "🟡 No documents indexed yet."
    )

else:

    st.warning(
        "🔴 FastAPI backend is not reachable."
    )


# ==================================================
# Upload section
# ==================================================

st.subheader("Upload Documents")

uploaded_files = st.file_uploader(
    "Choose PDF, DOCX, or TXT files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)


if uploaded_files:

    st.write(
        f"**{len(uploaded_files)} document(s) selected**"
    )

    for uploaded_file in uploaded_files:

        st.caption(
            f"📄 {uploaded_file.name}"
        )

    if st.button(
        "Index Documents",
        type="primary"
    ):

        multipart_files = []

        for uploaded_file in uploaded_files:

            multipart_files.append(
                (
                    "files",
                    (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                )
            )

        try:

            with st.spinner(
                "Extracting, chunking and indexing documents..."
            ):

                response = requests.post(
                    f"{API_URL}/upload",
                    files=multipart_files,
                    timeout=300
                )

            if response.status_code == 200:

                data = response.json()

                st.success(
                    f"✅ Indexed "
                    f"{len(data['documents_uploaded'])} document(s) | "
                    f"{data['chunks_created']} new chunks | "
                    f"{data['total_chunks']} total chunks"
                )

                st.rerun()

            else:

                st.error(
                    f"Upload failed "
                    f"({response.status_code}): "
                    f"{response.text}"
                )

        except requests.exceptions.Timeout:

            st.error(
                "Document indexing timed out."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI. "
                "Start the backend first."
            )

        except Exception as e:

            st.error(
                f"Unexpected upload error: {e}"
            )


# ==================================================
# Question answering
# ==================================================

st.divider()

st.subheader("Ask a Question")

question = st.text_input(
    "Ask something about your uploaded documents:"
)
compare_documents = st.checkbox(
    "Compare across uploaded documents",
    help="Forces document-aware comparison retrieval when you want a cross-document answer."
)


if st.button(
    "Ask",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "compare": compare_documents
                    },
                    timeout=180
                )

            if response.status_code == 200:

                data = response.json()

                # ----------------------------------
                # Answer
                # ----------------------------------

                st.subheader("Answer")

                st.write(
                    data["answer"]
                )

                # ----------------------------------
                # Sources
                # ----------------------------------

                st.subheader("Sources")

                sources = data.get(
                    "sources",
                    []
                )

                if sources:

                    seen = set()

                    for source in sources:

                        filename = source.get(
                            "source",
                            "Unknown source"
                        )

                        page = source.get(
                            "page"
                        )

                        filename = filename.replace(
                            "uploads\\",
                            ""
                        ).replace(
                            "uploads/",
                            ""
                        )

                        source_key = (
                            filename,
                            page
                        )

                        if source_key in seen:
                            continue

                        seen.add(
                            source_key
                        )

                        if page is not None:

                            st.write(
                                f"📄 **{filename}** "
                                f"— Page {page + 1}"
                            )

                        else:

                            st.write(
                                f"📄 **{filename}**"
                            )

                else:

                    st.caption(
                        "No source metadata available."
                    )

            else:

                st.error(
                    f"Query failed "
                    f"({response.status_code}): "
                    f"{response.text}"
                )

        except requests.exceptions.Timeout:

            st.error(
                "The backend took too long to respond."
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "Cannot connect to FastAPI."
            )

        except Exception as e:

            st.error(
                f"Unexpected error: {e}"
            )


# ==================================================
# Knowledge base
# ==================================================

st.divider()

st.subheader("Knowledge Base")

status_data = get_status()

if status_data and status_data["documents"]:

    st.write("Indexed documents:")

    for document in status_data["documents"]:

        st.write(
            f"📄 {document}"
        )

else:

    st.caption(
        "No documents currently indexed."
    )


# ==================================================
# Reset
# ==================================================

if st.button("Reset Documents"):

    try:

        response = requests.post(
            f"{API_URL}/reset",
            timeout=30
        )

        if response.status_code == 200:

            st.success(
                "✅ Knowledge base reset successfully."
            )

            st.rerun()

        else:

            st.error(
                f"Reset failed: {response.text}"
            )

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to FastAPI."
        )

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )