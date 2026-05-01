from __future__ import annotations

import os
import shutil
import stat
import time
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


def _make_writable_and_retry(function: object, path: str, exc_info: tuple[object, BaseException, object]) -> None:
    try:
        os.chmod(path, stat.S_IWRITE)
        function(path)
    except OSError:
        raise exc_info[1]


def recreate_collection(index_dir: Path) -> Collection:
    if index_dir.exists():
        for attempt in range(3):
            try:
                shutil.rmtree(index_dir, onerror=_make_writable_and_retry)
                break
            except PermissionError as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Could not recreate vector index at {index_dir}. "
                        "Windows is denying access to one or more Chroma index files. "
                        "Close any running chat/ask/index sessions using this index and retry, "
                        "or use a separate --index-dir when comparing embedding models."
                    ) from exc
                time.sleep(0.5)
    return get_collection(index_dir)
