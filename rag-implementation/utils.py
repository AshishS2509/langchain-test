from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import EMBEDDING_MODEL, LLM_MODEL, TEXT_FILE

def load_text_file() -> str:
    """
    Load the text file safely.

    Tries UTF-8 first.
    Falls back to UTF-8 with BOM.
    Finally falls back to latin-1.

    Returns:
        str
    """

    if not TEXT_FILE.exists():
        raise FileNotFoundError(
            f"Text file not found:\n{TEXT_FILE}"
        )

    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            return TEXT_FILE.read_text(encoding=encoding)
        except UnicodeDecodeError:
            pass

    raise RuntimeError(
        "Unable to read text file with supported encodings."
    )

load_text_file()

def get_embeddings() -> OllamaEmbeddings:
    """
    Create embedding model.
    """

    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
    )

def get_llm() -> ChatOllama:
    """
    Create chat model.

    Temperature 0 keeps answers deterministic.
    """

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )

def format_documents(documents) -> str:
    """
    Convert LangChain Documents into one string.

    This is the context that will be given
    to the language model.
    """

    return "\n\n".join(
        document.page_content
        for document in documents
    )

def print_documents(documents) -> None:
    """
    Pretty-print retrieved chunks.

    Useful while learning RAG.
    """

    print("\n")
    print("=" * 80)
    print("Retrieved Chunks")
    print("=" * 80)

    for index, doc in enumerate(documents, start=1):

        print(f"\nChunk {index}\n")

        print(doc.page_content.strip())

        print("\n" + "-" * 80)