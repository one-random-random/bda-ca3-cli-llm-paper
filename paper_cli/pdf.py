from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from paper_cli.models import PaperChunk


def ensure_pdf_exists(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {pdf_path}")


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
