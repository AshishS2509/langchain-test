from pathlib import Path

# -----------------------------
# Project Directories
# -----------------------------

BASE_DIR = Path(__file__).parent

TEXT_FILE = BASE_DIR / "data.txt"

CHROMA_DIR = BASE_DIR / "chroma_db"


# -----------------------------
# Ollama Models
# -----------------------------

EMBEDDING_MODEL = "qwen3-embedding:0.6b"

LLM_MODEL = "lfm2.5-thinking:latest"


# -----------------------------
# Chunk Settings
# -----------------------------

CHUNK_SIZE = 70

CHUNK_OVERLAP = 10


# -----------------------------
# Retriever
# -----------------------------

TOP_K = 3


# -----------------------------
# Chroma
# -----------------------------

COLLECTION_NAME = "notes"


# -----------------------------
# Ollama
# -----------------------------

