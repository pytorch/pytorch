from fsspec import AbstractFileSystem
from fsspec.utils import tokenize


class AbstractArchiveFileSystem(AbstractFileSystem):
    """
    A generic superclass for implementing Archive-based filesystems.

    Currently, it is shared amongst
    :class:`~fsspec.implementations.zip.ZipFileSystem`,
    :class:`~fsspec.implementations.libarchive.LibArchiveFileSystem` and
    :class:`~fsspec.implementations.tar.TarFileSystem`.
    """

    def __str__(self):
        return f"<Archive-like object {type(self).__name__} at {id(self)}>"

    __repr__ = __str__

    def ukey(self, path):
        return tokenize(path, self.fo, self.protocol)

    def _all_dirnames(self, paths):
        """Returns *all* directory names for each path in paths, including intermediate
        ones.

        Parameters
        ----------
        paths: Iterable of path strings
        """
        if len(paths) == 0:
            return set()

        dirnames = {self._parent(path) for path in paths} - {self.root_marker}
        return dirnames | self._all_dirnames(dirnames)

    def info(self, path, **kwargs):
        self._get_dirs()
        path = self._strip_protocol(path)
        if path in {"", "/"} and self.dir_cache:
            return {"name": "", "type": "directory", "size": 0}
        if path in self.dir_cache:
            return self.dir_cache[path]
        elif path + "/" in self.dir_cache:
            return self.dir_cache[path + "/"]
        else:
            raise FileNotFoundError(path)

    def ls(self, path, detail=True, **kwargs):
        self._get_dirs()
        paths = {}
        for p, f in self.dir_cache.items():
            p = p.rstrip("/")
            if "/" in p:
                root = p.rsplit("/", 1)[0]
            else:
                root = ""
            if root == path.rstrip("/"):
                paths[p] = f
            elif all(
                (a == b)
                for a, b in zip(path.split("/"), [""] + p.strip("/").split("/"))
            ):
                # root directory entry
                ppath = p.rstrip("/").split("/", 1)[0]
                if ppath not in paths:
                    out = {"name": ppath, "size": 0, "type": "directory"}
                    paths[ppath] = out
        if detail:
            out = sorted(paths.values(), key=lambda _: _["name"])
            return out
        else:
            return sorted(paths)
