from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_FOLDER = "/base_data"
PDF_NAME = "CLI Papaer -Research Landscape of Agentic AI and LLM - Apps, Challenges and Future Direction.pdf"
DEFAULT_PDF = APP_DIR / BASE_DATA_FOLDER / PDF_NAME
DEFAULT_INDEX_DIR = APP_DIR / ".paper_index"
COLLECTION_NAME = "agentic_ai_paper"
DEFAULT_CHAT_MODEL = "qwen3.5:4b" #llama3"  # "qwen3.5:4b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_WORDS = 260
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_MAX_DISTANCE = 0.5
