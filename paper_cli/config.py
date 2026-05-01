from __future__ import annotations

from pathlib import Path
import re


APP_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_FOLDER = "base_data"
PDF_NAME = "CLI Papaer -Research Landscape of Agentic AI and LLM - Apps, Challenges and Future Direction.pdf"
DEFAULT_PDF = APP_DIR / BASE_DATA_FOLDER / PDF_NAME
EMBED_MODEL_INDEX_NAMES = {
    "nomic-embed-text": "nomic",
    "mxbai-embed-large": "mxbai",
    "snowflake-arctic-embed": "snowflake",
}
COLLECTION_NAME = "agentic_ai_paper"
DEFAULT_CHAT_MODEL = "qwen3.5:4b"  #llama3"  
DEFAULT_EMBED_MODEL = "mxbai-embed-large"  # "mxbai-embed-large" # "snowflake-arctic-embed"
DEFAULT_INDEX_DIR = APP_DIR / f".paper_index_{EMBED_MODEL_INDEX_NAMES[DEFAULT_EMBED_MODEL]}"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_WORDS = 120
DEFAULT_CHUNK_OVERLAP = 30
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_DISTANCE = 0.5
RETRIEVAL_REFUSAL_MESSAGE = (
    "No relevant documents found. The paper I am referencing does not have relevant "
    "information for your question so I cannot answer."
)
LLM_REFUSAL_MESSAGE = "I cannot answer that from the provided paper."


def index_dir_for_embed_model(embed_model: str) -> Path:
    model_name = embed_model.split(":", maxsplit=1)[0]
    index_name = EMBED_MODEL_INDEX_NAMES.get(model_name)
    if index_name is None:
        index_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._-")
    return APP_DIR / f".paper_index_{index_name}"
