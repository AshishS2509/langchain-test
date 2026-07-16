"""
Index notes.txt into ChromaDB.

Run:

    python ingest.py

This script:

1. Checks Ollama
2. Checks required models
3. Loads notes.txt
4. Splits text into chunks
5. Deletes previous Chroma collection
6. Creates embeddings
7. Stores everything in ChromaDB
"""

import shutil

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
)

from utils import (
    get_embeddings,
    load_text_file,
)


def rebuild_database() -> None:
    """
    Completely rebuild the local vector database.

    For a beginner project this is much easier than trying
    to detect changed documents.
    """

    print("=" * 70)
    print("Local RAG Index")
    print("=" * 70)

    print("Loading text file...")

    text = load_text_file()

    if not text.strip():
        raise RuntimeError("notes.txt is empty.")

    print("Splitting document...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    documents = splitter.create_documents([text])

    print(f"Created {len(documents)} chunk(s).")

    # --------------------------------------------------------
    # Remove previous database
    # --------------------------------------------------------

    if CHROMA_DIR.exists():

        print("Removing previous database...")

        shutil.rmtree(CHROMA_DIR)

    print("Creating embedding model...")

    embeddings = get_embeddings()

    print("Creating Chroma database...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    print("Generating embeddings...")

    vectorstore.add_documents(documents)

    print()

    print("=" * 70)
    print("Index Complete")
    print("=" * 70)

    print(f"Chunks indexed : {len(documents)}")
    print(f"Database       : {CHROMA_DIR}")
    print()


if __name__ == "__main__":
    rebuild_database()