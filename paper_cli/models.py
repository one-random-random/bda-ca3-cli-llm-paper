from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperChunk:
    chunk_id: str
    text: str
    page: int
    chunk_number: int
