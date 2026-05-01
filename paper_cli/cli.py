from __future__ import annotations

import argparse

from paper_cli.commands import ask, chat, index_paper
from paper_cli.config import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_WORDS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_PDF,
    DEFAULT_TOP_K,
)
from paper_cli.ui import console


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask questions of a local Ollama model grounded only in one PDF paper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Extract and index the PDF paper.")
    index_parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to the source PDF.")
    index_parser.add_argument("--index-dir", default=None, help="Persistent Chroma index directory. Defaults to .paper_index_<embed model>.")
    index_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    index_parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS, help="Words per text chunk.")
    index_parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlapping words between chunks.")
    index_parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size.")
    index_parser.set_defaults(func=index_paper)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive paper-grounded chat.")
    chat_parser.add_argument("--index-dir", default=None, help="Persistent Chroma index directory. Defaults to .paper_index_<embed model>.")
    chat_parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Ollama chat model.")
    chat_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    chat_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of paper chunks to retrieve.")
    chat_parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help="Cosine distance cutoff. Lower is stricter; increase if valid questions are refused.",
    )
    chat_parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    chat_parser.add_argument("--num-predict", type=int, default=300, help="Maximum tokens to generate.")
    chat_parser.add_argument("--debug", action="store_true", help="Show retrieval distances.")
    chat_parser.set_defaults(func=chat)

    ask_parser = subparsers.add_parser("ask", help="Ask one question and print one answer.")
    ask_parser.add_argument("question", help="Question to ask about the indexed paper.")
    ask_parser.add_argument("--index-dir", default=None, help="Persistent Chroma index directory. Defaults to .paper_index_<embed model>.")
    ask_parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Ollama chat model.")
    ask_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of paper chunks to retrieve.")
    ask_parser.add_argument(
        "--max-distance",
        type=float,
        default=DEFAULT_MAX_DISTANCE,
        help="Cosine distance cutoff. Lower is stricter; increase if valid questions are refused.",
    )
    ask_parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    ask_parser.add_argument("--num-predict", type=int, default=300, help="Maximum tokens to generate.")
    ask_parser.add_argument("--debug", action="store_true", help="Show retrieval distances.")
    ask_parser.set_defaults(func=ask)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        return 1
    return 0
