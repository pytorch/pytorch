from __future__ import annotations

import logging
from datetime import datetime, timezone
from errno import ENOTEMPTY
from io import BytesIO
from pathlib import PurePath, PureWindowsPath
from typing import Any, ClassVar

from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from fsspec.utils import stringify_path

logger = logging.getLogger("fsspec.memoryfs")


class MemoryFileSystem(AbstractFileSystem):
    """A filesystem based on a dict of BytesIO objects

    This is a global filesystem so instances of this class all point to the same
    in memory filesystem.
    """

    store: ClassVar[dict[str, Any]] = {}  # global, do not overwrite!
    pseudo_dirs = [""]  # global, do not overwrite!
    protocol = "memory"
    root_marker = "/"

    @classmethod
    def _strip_protocol(cls, path):
        if isinstance(path, PurePath):
            if isinstance(path, PureWindowsPath):
                return LocalFileSystem._strip_protocol(path)
            else:
                path = stringify_path(path)

        path = path.removeprefix("memory://")
        if "::" in path or "://" in path:
            return path.rstrip("/")
        path = path.lstrip("/").rstrip("/")
        return "/" + path if path else ""

    def ls(self, path, detail=True, **kwargs):
        path = self._strip_protocol(path)
        if path in self.store:
            # there is a key with this exact name
            if not detail:
                return [path]
            return [
                {
                    "name": path,
                    "size": self.store[path].size,
                    "type": "file",
                    "created": self.store[path].created.timestamp(),
                }
            ]
        paths = set()
        starter = path + "/"
        out = []
        for p2 in tuple(self.store):
            if p2.startswith(starter):
                if "/" not in p2[len(starter) :]:
                    # exact child
                    out.append(
                        {
                            "name": p2,
                            "size": self.store[p2].size,
                            "type": "file",
                            "created": self.store[p2].created.timestamp(),
                        }
                    )
                elif len(p2) > len(starter):
                    # implied child directory
                    ppath = starter + p2[len(starter) :].split("/", 1)[0]
                    if ppath not in paths:
                        out = out or []
                        out.append(
                            {
                                "name": ppath,
                                "size": 0,
                                "type": "directory",
                            }
                        )
                        paths.add(ppath)
        for p2 in self.pseudo_dirs:
            if p2.startswith(starter):
                if "/" not in p2[len(starter) :]:
                    # exact child pdir
                    if p2 not in paths:
                        out.append({"name": p2, "size": 0, "type": "directory"})
                        paths.add(p2)
                else:
                    # directory implied by deeper pdir
                    ppath = starter + p2[len(starter) :].split("/", 1)[0]
                    if ppath not in paths:
                        out.append({"name": ppath, "size": 0, "type": "directory"})
                        paths.add(ppath)
        if not out:
            if path in self.pseudo_dirs:
                # empty dir
                return []
            raise FileNotFoundError(path)
        if detail:
            return out
        return sorted([f["name"] for f in out])

    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if path in self.store or path in self.pseudo_dirs:
            raise FileExistsError(path)
        if self._parent(path).strip("/") and self.isfile(self._parent(path)):
            raise NotADirectoryError(self._parent(path))
        if create_parents and self._parent(path).strip("/"):
            try:
                self.mkdir(self._parent(path), create_parents, **kwargs)
            except FileExistsError:
                pass
        if path and path not in self.pseudo_dirs:
            self.pseudo_dirs.append(path)

    def makedirs(self, path, exist_ok=False):
        try:
            self.mkdir(path, create_parents=True)
        except FileExistsError:
            if not exist_ok:
                raise

    def pipe_file(self, path, value, mode="overwrite", **kwargs):
        """Set the bytes of given file

        Avoids copies of the data if possible
        """
        mode = "xb" if mode == "create" else "wb"
        self.open(path, mode=mode, data=value)

    def rmdir(self, path):
        path = self._strip_protocol(path)
        if path == "":
            # silently avoid deleting FS root
            return
        if path in self.pseudo_dirs:
            if not self.ls(path):
                self.pseudo_dirs.remove(path)
            else:
                raise OSError(ENOTEMPTY, "Directory not empty", path)
        else:
            raise FileNotFoundError(path)

    def info(self, path, **kwargs):
        logger.debug("info: %s", path)
        path = self._strip_protocol(path)
        if path in self.pseudo_dirs or any(
            p.startswith(path + "/") for p in list(self.store) + self.pseudo_dirs
        ):
            return {
                "name": path,
                "size": 0,
                "type": "directory",
            }
        elif path in self.store:
            filelike = self.store[path]
            return {
                "name": path,
                "size": filelike.size,
                "type": "file",
                "created": getattr(filelike, "created", None),
            }
        else:
            raise FileNotFoundError(path)

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if "x" in mode and self.exists(path):
            raise FileExistsError
        if path in self.pseudo_dirs:
            raise IsADirectoryError(path)
        parent = path
        while len(parent) > 1:
            parent = self._parent(parent)
            if self.isfile(parent):
                raise FileExistsError(parent)
        if mode in ["rb", "ab", "r+b", "a+b"]:
            if path in self.store:
                f = self.store[path]
                if "a" in mode:
                    # position at the end of file
                    f.seek(0, 2)
                else:
                    # position at the beginning of file
                    f.seek(0)
                return f
            else:
                raise FileNotFoundError(path)
        elif mode in {"wb", "w+b", "xb", "x+b"}:
            if "x" in mode and self.exists(path):
                raise FileExistsError
            m = MemoryFile(self, path, kwargs.get("data"))
            if not self._intrans:
                m.commit()
            return m
        else:
            name = self.__class__.__name__
            raise ValueError(f"unsupported file mode for {name}: {mode!r}")

    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1)
        path2 = self._strip_protocol(path2)
        if self.isfile(path1):
            self.store[path2] = MemoryFile(
                self, path2, self.store[path1].getvalue()
            )  # implicit copy
        elif self.isdir(path1):
            if path2 not in self.pseudo_dirs:
                self.pseudo_dirs.append(path2)
        else:
            raise FileNotFoundError(path1)

    def cat_file(self, path, start=None, end=None, **kwargs):
        logger.debug("cat: %s", path)
        path = self._strip_protocol(path)
        try:
            return bytes(self.store[path].getbuffer()[start:end])
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def _rm(self, path):
        path = self._strip_protocol(path)
        try:
            del self.store[path]
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def modified(self, path):
        path = self._strip_protocol(path)
        try:
            return self.store[path].modified
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def created(self, path):
        path = self._strip_protocol(path)
        try:
            return self.store[path].created
        except KeyError as e:
            raise FileNotFoundError(path) from e

    def isfile(self, path):
        path = self._strip_protocol(path)
        return path in self.store

    def rm(self, path, recursive=False, maxdepth=None):
        if isinstance(path, str):
            path = self._strip_protocol(path)
        else:
            path = [self._strip_protocol(p) for p in path]
        paths = self.expand_path(path, recursive=recursive, maxdepth=maxdepth)
        for p in reversed(paths):
            if self.isfile(p):
                self.rm_file(p)
            # If the expanded path doesn't exist, it is only because the expanded
            # path was a directory that does not exist in self.pseudo_dirs. This
            # is possible if you directly create files without making the
            # directories first.
            elif not self.exists(p):
                continue
            else:
                self.rmdir(p)


class MemoryFile(BytesIO):
    """A BytesIO which can't close and works as a context manager

    Can initialise with data. Each path should only be active once at any moment.

    No need to provide fs, path if auto-committing (default)
    """

    def __init__(self, fs=None, path=None, data=None):
        logger.debug("open file %s", path)
        self.fs = fs
        self.path = path
        self.created = datetime.now(tz=timezone.utc)
        self.modified = datetime.now(tz=timezone.utc)
        if data:
            super().__init__(data)
            self.seek(0)

    @property
    def size(self):
        return self.getbuffer().nbytes

    def __enter__(self):
        return self

    def close(self):
        pass

    def discard(self):
        pass

    def commit(self):
        self.fs.store[self.path] = self
        self.modified = datetime.now(tz=timezone.utc)
