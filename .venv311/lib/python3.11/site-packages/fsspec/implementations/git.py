import os

import pygit2

from fsspec.spec import AbstractFileSystem

from .memory import MemoryFile


class GitFileSystem(AbstractFileSystem):
    """Browse the files of a local git repo at any hash/tag/branch

    (experimental backend)
    """

    root_marker = ""
    cachable = True

    def __init__(self, path=None, fo=None, ref=None, **kwargs):
        """

        Parameters
        ----------
        path: str (optional)
            Local location of the repo (uses current directory if not given).
            May be deprecated in favour of ``fo``. When used with a higher
            level function such as fsspec.open(), may be of the form
            "git://[path-to-repo[:]][ref@]path/to/file" (but the actual
            file path should not contain "@" or ":").
        fo: str (optional)
            Same as ``path``, but passed as part of a chained URL. This one
            takes precedence if both are given.
        ref: str (optional)
            Reference to work with, could be a hash, tag or branch name. Defaults
            to current working tree. Note that ``ls`` and ``open`` also take hash,
            so this becomes the default for those operations
        kwargs
        """
        super().__init__(**kwargs)
        self.repo = pygit2.Repository(fo or path or os.getcwd())
        self.ref = ref or "master"

    @classmethod
    def _strip_protocol(cls, path):
        path = super()._strip_protocol(path).lstrip("/")
        if ":" in path:
            path = path.split(":", 1)[1]
        if "@" in path:
            path = path.split("@", 1)[1]
        return path.lstrip("/")

    def _path_to_object(self, path, ref):
        comm, ref = self.repo.resolve_refish(ref or self.ref)
        parts = path.split("/")
        tree = comm.tree
        for part in parts:
            if part and isinstance(tree, pygit2.Tree):
                if part not in tree:
                    raise FileNotFoundError(path)
                tree = tree[part]
        return tree

    @staticmethod
    def _get_kwargs_from_urls(path):
        path = path.removeprefix("git://")
        out = {}
        if ":" in path:
            out["path"], path = path.split(":", 1)
        if "@" in path:
            out["ref"], path = path.split("@", 1)
        return out

    @staticmethod
    def _object_to_info(obj, path=None):
        # obj.name and obj.filemode are None for the root tree!
        is_dir = isinstance(obj, pygit2.Tree)
        return {
            "type": "directory" if is_dir else "file",
            "name": (
                "/".join([path, obj.name or ""]).lstrip("/") if path else obj.name
            ),
            "hex": str(obj.id),
            "mode": "100644" if obj.filemode is None else f"{obj.filemode:o}",
            "size": 0 if is_dir else obj.size,
        }

    def ls(self, path, detail=True, ref=None, **kwargs):
        tree = self._path_to_object(self._strip_protocol(path), ref)
        return [
            GitFileSystem._object_to_info(obj, path)
            if detail
            else GitFileSystem._object_to_info(obj, path)["name"]
            for obj in (tree if isinstance(tree, pygit2.Tree) else [tree])
        ]

    def info(self, path, ref=None, **kwargs):
        tree = self._path_to_object(self._strip_protocol(path), ref)
        return GitFileSystem._object_to_info(tree, path)

    def ukey(self, path, ref=None):
        return self.info(path, ref=ref)["hex"]

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        ref=None,
        **kwargs,
    ):
        obj = self._path_to_object(path, ref or self.ref)
        return MemoryFile(data=obj.data)
