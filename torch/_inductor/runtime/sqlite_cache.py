"""Optional SQLite storage for local autotuning records and Python source."""

from __future__ import annotations

import atexit
import functools
import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator


def sqlite_cache_enabled() -> bool:
    backend = os.environ.get("TORCHINDUCTOR_CACHE_BACKEND", "file")
    if backend not in ("file", "sqlite"):
        raise ValueError(f"Unknown TORCHINDUCTOR_CACHE_BACKEND: {backend!r}")
    return backend == "sqlite"


class SQLiteCache:
    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.database = os.path.join(self.root, "inductor-cache-v1.sqlite3")

    @contextmanager
    def _connect(self, *, initialize: bool = True) -> Iterator[sqlite3.Connection]:
        import sqlite3

        connection = None
        try:
            if not initialize:
                # Open only an existing database, while allowing hot-journal recovery.
                connection = sqlite3.connect(
                    Path(self.database).as_uri() + "?mode=rw", uri=True, timeout=30
                )
            else:
                os.makedirs(self.root, exist_ok=True)
                connection = sqlite3.connect(self.database, timeout=30)
            with connection:
                if initialize:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS entries ("
                        "namespace TEXT NOT NULL, key TEXT NOT NULL, data BLOB NOT NULL, "
                        "PRIMARY KEY (namespace, key)) WITHOUT ROWID"
                    )
                yield connection
        except sqlite3.Error as error:
            raise RuntimeError(
                f"SQLite cache failure at {self.database}: {error}. "
                "Use writable local storage for compilation; stop all users before clearing a corrupt cache."
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def get(self, namespace: str, key: str) -> bytes | None:
        if not os.path.exists(self.database):
            return None
        with self._connect(initialize=False) as connection:
            # Another process may have opened the file but not created its schema yet.
            if (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entries'"
                ).fetchone()
                is None
            ):
                return None
            row = connection.execute(
                "SELECT data FROM entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        return None if row is None else row[0]

    def put(
        self, namespace: str, key: str, data: bytes, *, overwrite: bool = True
    ) -> None:
        statement = (
            "INSERT OR REPLACE INTO entries VALUES (?, ?, ?)"
            if overwrite
            else "INSERT OR IGNORE INTO entries VALUES (?, ?, ?)"
        )
        with self._connect() as connection:
            connection.execute(statement, (namespace, key, data))

    def delete(self, namespace: str, key: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM entries WHERE namespace = ? AND key = ?", (namespace, key)
            )

    def materialize_python(self, key: str) -> str:
        if not key or os.path.basename(key) != key or key in (".", ".."):
            raise ValueError(f"Invalid Python cache key: {key!r}")
        root = _runtime_dir(os.getpid(), self.root)
        directory = os.path.join(root, key[1:3])
        path = os.path.join(directory, f"{key}.py")
        if not os.path.exists(path):
            source = self.get("python", key)
            if source is None:
                raise FileNotFoundError(
                    f"Python cache entry {key} missing from {self.database}"
                )
            os.makedirs(directory, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=directory)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(source)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return path

    def autotune_key(self, path: str) -> str:
        # Python materializations retain the usual <prefix>/<hash> layout so
        # bundled artifacts and records survive process/runtime restarts.
        runtime_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        prefix = _runtime_prefix(self.root)
        if os.path.basename(runtime_root).startswith(prefix):
            return os.path.join(
                os.path.basename(os.path.dirname(path)), os.path.basename(path)
            )
        return os.path.relpath(os.path.abspath(path), self.root)


def _runtime_prefix(root: str) -> str:
    digest = hashlib.sha256(root.encode()).hexdigest()[:16]
    return f"torchinductor-sqlite-{digest}-"


@functools.lru_cache(None)
def _runtime_dir(pid: int, root: str) -> str:
    directory = tempfile.mkdtemp(prefix=_runtime_prefix(root))

    def cleanup() -> None:
        if os.getpid() == pid:
            shutil.rmtree(directory, ignore_errors=True)

    atexit.register(cleanup)
    return directory


def local_cache() -> SQLiteCache:
    from .runtime_utils import cache_dir

    return SQLiteCache(cache_dir())
