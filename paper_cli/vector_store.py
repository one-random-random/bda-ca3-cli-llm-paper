from __future__ import annotations

import shutil
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from paper_cli.config import COLLECTION_NAME


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
