from __future__ import annotations

import os
import pickle
import time
from typing import TYPE_CHECKING

from fsspec.utils import atomic_write

try:
    import ujson as json
except ImportError:
    if not TYPE_CHECKING:
        import json

if TYPE_CHECKING:
    from typing import Any, Dict, Iterator, Literal

    from typing_extensions import TypeAlias

    from .cached import CachingFileSystem

    Detail: TypeAlias = Dict[str, Any]


class CacheMetadata:
    """Cache metadata.

    All reading and writing of cache metadata is performed by this class,
    accessing the cached files and blocks is not.

    Metadata is stored in a single file per storage directory in JSON format.
    For backward compatibility, also reads metadata stored in pickle format
    which is converted to JSON when next saved.
    """

    def __init__(self, storage: list[str]):
        """

        Parameters
        ----------
        storage: list[str]
            Directories containing cached files, must be at least one. Metadata
            is stored in the last of these directories by convention.
        """
        if not storage:
            raise ValueError("CacheMetadata expects at least one storage location")

        self._storage = storage
        self.cached_files: list[Detail] = [{}]

        # Private attribute to force saving of metadata in pickle format rather than
        # JSON for use in tests to confirm can read both pickle and JSON formats.
        self._force_save_pickle = False

    def _load(self, fn: str) -> Detail:
        """Low-level function to load metadata from specific file"""
        try:
            with open(fn, "r") as f:
                loaded = json.load(f)
        except ValueError:
            with open(fn, "rb") as f:
                loaded = pickle.load(f)
        for c in loaded.values():
            if isinstance(c.get("blocks"), list):
                c["blocks"] = set(c["blocks"])
        return loaded

    def _save(self, metadata_to_save: Detail, fn: str) -> None:
        """Low-level function to save metadata to specific file"""
        if self._force_save_pickle:
            with atomic_write(fn) as f:
                pickle.dump(metadata_to_save, f)
        else:
            with atomic_write(fn, mode="w") as f:
                json.dump(metadata_to_save, f)

    def _scan_locations(
        self, writable_only: bool = False
    ) -> Iterator[tuple[str, str, bool]]:
        """Yield locations (filenames) where metadata is stored, and whether
        writable or not.

        Parameters
        ----------
        writable: bool
            Set to True to only yield writable locations.

        Returns
        -------
        Yields (str, str, bool)
        """
        n = len(self._storage)
        for i, storage in enumerate(self._storage):
            writable = i == n - 1
            if writable_only and not writable:
                continue
            yield os.path.join(storage, "cache"), storage, writable

    def check_file(
        self, path: str, cfs: CachingFileSystem | None
    ) -> Literal[False] | tuple[Detail, str]:
        """If path is in cache return its details, otherwise return ``False``.

        If the optional CachingFileSystem is specified then it is used to
        perform extra checks to reject possible matches, such as if they are
        too old.
        """
        for (fn, base, _), cache in zip(self._scan_locations(), self.cached_files):
            if path not in cache:
                continue
            detail = cache[path].copy()

            if cfs is not None:
                if cfs.check_files and detail["uid"] != cfs.fs.ukey(path):
                    # Wrong file as determined by hash of file properties
                    continue
                if cfs.expiry and time.time() - detail["time"] > cfs.expiry:
                    # Cached file has expired
                    continue

            fn = os.path.join(base, detail["fn"])
            if os.path.exists(fn):
                return detail, fn
        return False

    def clear_expired(self, expiry_time: int) -> tuple[list[str], bool]:
        """Remove expired metadata from the cache.

        Returns names of files corresponding to expired metadata and a boolean
        flag indicating whether the writable cache is empty. Caller is
        responsible for deleting the expired files.
        """
        expired_files = []
        for path, detail in self.cached_files[-1].copy().items():
            if time.time() - detail["time"] > expiry_time:
                fn = detail.get("fn", "")
                if not fn:
                    raise RuntimeError(
                        f"Cache metadata does not contain 'fn' for {path}"
                    )
                fn = os.path.join(self._storage[-1], fn)
                expired_files.append(fn)
                self.cached_files[-1].pop(path)

        if self.cached_files[-1]:
            cache_path = os.path.join(self._storage[-1], "cache")
            self._save(self.cached_files[-1], cache_path)

        writable_cache_empty = not self.cached_files[-1]
        return expired_files, writable_cache_empty

    def load(self) -> None:
        """Load all metadata from disk and store in ``self.cached_files``"""
        cached_files = []
        for fn, _, _ in self._scan_locations():
            if os.path.exists(fn):
                # TODO: consolidate blocks here
                cached_files.append(self._load(fn))
            else:
                cached_files.append({})
        self.cached_files = cached_files or [{}]

    def on_close_cached_file(self, f: Any, path: str) -> None:
        """Perform side-effect actions on closing a cached file.

        The actual closing of the file is the responsibility of the caller.
        """
        # File must be writeble, so in self.cached_files[-1]
        c = self.cached_files[-1][path]
        if c["blocks"] is not True and len(c["blocks"]) * f.blocksize >= f.size:
            c["blocks"] = True

    def pop_file(self, path: str) -> str | None:
        """Remove metadata of cached file.

        If path is in the cache, return the filename of the cached file,
        otherwise return ``None``.  Caller is responsible for deleting the
        cached file.
        """
        details = self.check_file(path, None)
        if not details:
            return None
        _, fn = details
        if fn.startswith(self._storage[-1]):
            self.cached_files[-1].pop(path)
            self.save()
        else:
            raise PermissionError(
                "Can only delete cached file in last, writable cache location"
            )
        return fn

    def save(self) -> None:
        """Save metadata to disk"""
        for (fn, _, writable), cache in zip(self._scan_locations(), self.cached_files):
            if not writable:
                continue

            if os.path.exists(fn):
                cached_files = self._load(fn)
                for k, c in cached_files.items():
                    if k in cache:
                        if c["blocks"] is True or cache[k]["blocks"] is True:
                            c["blocks"] = True
                        else:
                            # self.cached_files[*][*]["blocks"] must continue to
                            # point to the same set object so that updates
                            # performed by MMapCache are propagated back to
                            # self.cached_files.
                            blocks = cache[k]["blocks"]
                            blocks.update(c["blocks"])
                            c["blocks"] = blocks
                        c["time"] = max(c["time"], cache[k]["time"])
                        c["uid"] = cache[k]["uid"]

                # Files can be added to cache after it was written once
                for k, c in cache.items():
                    if k not in cached_files:
                        cached_files[k] = c
            else:
                cached_files = cache
            cache = {k: v.copy() for k, v in cached_files.items()}
            for c in cache.values():
                if isinstance(c["blocks"], set):
                    c["blocks"] = list(c["blocks"])
            self._save(cache, fn)
            self.cached_files[-1] = cached_files

    def update_file(self, path: str, detail: Detail) -> None:
        """Update metadata for specific file in memory, do not save"""
        self.cached_files[-1][path] = detail
