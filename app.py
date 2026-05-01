from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import ollama
from chromadb.api.models.Collection import Collection
from pypdf import PdfReader
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt


APP_DIR = Path(__file__).resolve().parent
BASE_DATA_FOLDER = "/base_data"
DEFAULT_PDF = APP_DIR / BASE_DATA_FOLDER / "CLI Papaer -Research Landscape of Agentic AI and LLM - Apps, Challenges and Future Direction.pdf"
DEFAULT_INDEX_DIR = APP_DIR / ".paper_index"
COLLECTION_NAME = "agentic_ai_paper"
DEFAULT_CHAT_MODEL = "llama3" # "qwen3.5:4b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_WORDS = 260
DEFAULT_CHUNK_OVERLAP = 60
DEFAULT_MAX_DISTANCE = 0.8

console = Console()


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    text: str
    page: int
    chunk_number: int


def ensure_pdf_exists(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")


def get_collection(index_dir: Path) -> Collection:
    client = chromadb.PersistentClient(path=str(index_dir))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def recreate_collection(index_dir: Path) -> Collection:
    if index_dir.exists():
        shutil.rmtree(index_dir)
    return get_collection(index_dir)


def extract_pdf_pages(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        cleaned = " ".join(text.split())
        if cleaned:
            pages.append((page_index, cleaned))
    return pages


def chunk_words(words: list[str], chunk_words_count: int, overlap: int) -> Iterable[list[str]]:
    if chunk_words_count <= 0:
        raise ValueError("chunk_words_count must be greater than zero")
    if overlap < 0 or overlap >= chunk_words_count:
        raise ValueError("overlap must be zero or less than chunk_words")

    start = 0
    while start < len(words):
        end = min(start + chunk_words_count, len(words))
        yield words[start:end]
        if end == len(words):
            break
        start = end - overlap


def build_chunks(pdf_path: Path, chunk_words_count: int, overlap: int) -> list[PaperChunk]:
    pages = extract_pdf_pages(pdf_path)
    chunks: list[PaperChunk] = []
    for page_number, page_text in pages:
        words = page_text.split()
        for chunk_index, word_chunk in enumerate(chunk_words(words, chunk_words_count, overlap), start=1):
            text = " ".join(word_chunk)
            chunk_id = f"{pdf_path.stem}-p{page_number}-c{chunk_index}"
            chunks.append(
                PaperChunk(
                    chunk_id=chunk_id,
                    text=text,
                    page=page_number,
                    chunk_number=chunk_index,
                )
            )
    return chunks


def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        response = ollama.embed(model=model, input=texts)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create embeddings with Ollama model '{model}'. "
            f"Run: ollama pull {model}"
        ) from exc

    embeddings = response.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Ollama returned no embeddings for model '{model}'.")
    return embeddings


def index_paper(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    ensure_pdf_exists(pdf_path)

    console.print(Panel.fit(f"Indexing\n[bold]{pdf_path.name}[/bold]", title="Paper CLI"))
    chunks = build_chunks(pdf_path, args.chunk_words, args.chunk_overlap)
    if not chunks:
        raise RuntimeError("No extractable text was found in the PDF.")

    collection = recreate_collection(index_dir)
    batch_size = args.batch_size
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        embeddings = embed_texts(args.embed_model, texts)
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "pdf_file": pdf_path.name,
                    "page": chunk.page,
                    "chunk_number": chunk.chunk_number,
                    "source_text": chunk.text,
                }
                for chunk in batch
            ],
        )
        console.print(f"Indexed {min(start + batch_size, total)}/{total} chunks")

    console.print(f"[green]Done.[/green] Vector index stored at: {index_dir}")


def format_context(documents: list[str], metadatas: list[dict]) -> tuple[str, list[int]]:
    context_blocks: list[str] = []
    pages: list[int] = []
    for number, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        page = int(metadata.get("page", 0))
        pages.append(page)
        context_blocks.append(f"[Excerpt {number} | page {page}]\n{document}")
    return "\n\n".join(context_blocks), sorted(set(pages))


def retrieve_context(
    collection: Collection,
    question: str,
    embed_model: str,
    top_k: int,
    max_distance: float,
) -> tuple[str | None, list[int], list[float]]:
    query_embedding = embed_texts(embed_model, [question])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not documents or not distances:
        return None, [], []
    if float(distances[0]) > max_distance:
        return None, [], [float(distance) for distance in distances]

    context, pages = format_context(documents, metadatas)
    return context, pages, [float(distance) for distance in distances]


def answer_question(
    question: str,
    context: str,
    pages: list[int],
    chat_model: str,
    temperature: float,
    num_predict: int,
) -> str:
    system_prompt = (
        "You are a careful research assistant. Answer the user's question using only the "
        "paper excerpts provided in the context. Do not use outside knowledge. If the "
        "context does not contain enough evidence, answer exactly: "
        "'I cannot answer that from the provided paper.' with no citations."
        "When you find an answer in the paper include page citations in the "
        "form (p. 3) or (pp. 3, 5) for every factual answer."
    )
    user_prompt = (
        f"Paper excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the excerpts. Available source pages: {', '.join(map(str, pages))}."
    )

    try:
        response = ollama.chat(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            think=False,
            options={"temperature": temperature, "num_predict": num_predict},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not chat with Ollama model '{chat_model}'. Run: ollama pull {chat_model}"
        ) from exc

    return response["message"]["content"].strip()


def chat(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).expanduser().resolve()
    if not index_dir.exists():
        raise RuntimeError("No paper index found. Run: python app.py index")

    collection = get_collection(index_dir)
    if collection.count() == 0:
        raise RuntimeError("The paper index is empty. Run: python app.py index")

    console.print(
        Panel.fit(
            "Ask questions about the indexed paper.\nType [bold]exit[/bold], [bold]quit[/bold], or press Ctrl+C to leave.",
            title="Paper-Grounded LLM CLI",
        )
    )

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]Question[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye.")
            return

        if question.lower() in {"exit", "quit", "q"}:
            console.print("Goodbye.")
            return
        if not question:
            continue

        try:
            context, pages, distances = retrieve_context(
                collection=collection,
                question=question,
                embed_model=args.embed_model,
                top_k=args.top_k,
                max_distance=args.max_distance,
            )
            if context is None:
                console.print(
                    Panel(
                        "I cannot answer that from the provided paper.",
                        title="Answer",
                        border_style="yellow",
                    )
                )
                if args.debug and distances:
                    console.print(f"[dim]Retrieval distances: {distances}[/dim]")
                continue

            answer = answer_question(
                question=question,
                context=context,
                pages=pages,
                chat_model=args.chat_model,
                temperature=args.temperature,
                num_predict=args.num_predict,
            )
            console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
            console.print(f"[dim]Source pages: {', '.join(f'p. {page}' for page in pages)}[/dim]")
            if args.debug:
                console.print(f"[dim]Retrieval distances: {distances}[/dim]")
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")


def ask(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).expanduser().resolve()
    if not index_dir.exists():
        raise RuntimeError("No paper index found. Run: python app.py index")

    collection = get_collection(index_dir)
    if collection.count() == 0:
        raise RuntimeError("The paper index is empty. Run: python app.py index")

    context, pages, distances = retrieve_context(
        collection=collection,
        question=args.question,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_distance=args.max_distance,
    )
    if context is None:
        console.print("I cannot answer that from the provided paper.")
        if args.debug and distances:
            console.print(f"[dim]Retrieval distances: {distances}[/dim]")
        return

    answer = answer_question(
        question=args.question,
        context=context,
        pages=pages,
        chat_model=args.chat_model,
        temperature=args.temperature,
        num_predict=args.num_predict,
    )
    console.print(Markdown(answer))
    console.print(f"[dim]Source pages: {', '.join(f'p. {page}' for page in pages)}[/dim]")
    if args.debug:
        console.print(f"[dim]Retrieval distances: {distances}[/dim]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask questions of a local Ollama model grounded only in one PDF paper."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Extract and index the PDF paper.")
    index_parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to the source PDF.")
    index_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Persistent Chroma index directory.")
    index_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    index_parser.add_argument("--chunk-words", type=int, default=DEFAULT_CHUNK_WORDS, help="Words per text chunk.")
    index_parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Overlapping words between chunks.")
    index_parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")
    index_parser.set_defaults(func=index_paper)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive paper-grounded chat.")
    chat_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Persistent Chroma index directory.")
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
    ask_parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="Persistent Chroma index directory.")
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


if __name__ == "__main__":
    sys.exit(main())
