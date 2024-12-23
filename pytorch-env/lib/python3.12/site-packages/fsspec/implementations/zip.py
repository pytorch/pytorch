import zipfile

import fsspec
from fsspec.archive import AbstractArchiveFileSystem


class ZipFileSystem(AbstractArchiveFileSystem):
    """Read/Write contents of ZIP archive as a file-system

    Keeps file object open while instance lives.

    This class is pickleable, but not necessarily thread-safe
    """

    root_marker = ""
    protocol = "zip"
    cachable = False

    def __init__(
        self,
        fo="",
        mode="r",
        target_protocol=None,
        target_options=None,
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
        compresslevel=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fo: str or file-like
            Contains ZIP, and must exist. If a str, will fetch file using
            :meth:`~fsspec.open_files`, which must return one file exactly.
        mode: str
            Accept: "r", "w", "a"
        target_protocol: str (optional)
            If ``fo`` is a string, this value can be used to override the
            FS protocol inferred from a URL
        target_options: dict (optional)
            Kwargs passed when instantiating the target FS, if ``fo`` is
            a string.
        compression, allowZip64, compresslevel: passed to ZipFile
            Only relevant when creating a ZIP
        """
        super().__init__(self, **kwargs)
        if mode not in set("rwa"):
            raise ValueError(f"mode '{mode}' no understood")
        self.mode = mode
        if isinstance(fo, str):
            if mode == "a":
                m = "r+b"
            else:
                m = mode + "b"
            fo = fsspec.open(
                fo, mode=m, protocol=target_protocol, **(target_options or {})
            )
        self.force_zip_64 = allowZip64
        self.of = fo
        self.fo = fo.__enter__()  # the whole instance is a context
        self.zip = zipfile.ZipFile(
            self.fo,
            mode=mode,
            compression=compression,
            allowZip64=allowZip64,
            compresslevel=compresslevel,
        )
        self.dir_cache = None

    @classmethod
    def _strip_protocol(cls, path):
        # zip file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def __del__(self):
        if hasattr(self, "zip"):
            self.close()
            del self.zip

    def close(self):
        """Commits any write changes to the file. Done on ``del`` too."""
        self.zip.close()

    def _get_dirs(self):
        if self.dir_cache is None or self.mode in set("wa"):
            # when writing, dir_cache is always in the ZipFile's attributes,
            # not read from the file.
            files = self.zip.infolist()
            self.dir_cache = {
                dirname.rstrip("/"): {
                    "name": dirname.rstrip("/"),
                    "size": 0,
                    "type": "directory",
                }
                for dirname in self._all_dirnames(self.zip.namelist())
            }
            for z in files:
                f = {s: getattr(z, s, None) for s in zipfile.ZipInfo.__slots__}
                f.update(
                    {
                        "name": z.filename.rstrip("/"),
                        "size": z.file_size,
                        "type": ("directory" if z.is_dir() else "file"),
                    }
                )
                self.dir_cache[f["name"]] = f

    def pipe_file(self, path, value, **kwargs):
        # override upstream, because we know the exact file size in this case
        self.zip.writestr(path, value, **kwargs)

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
        if "r" in mode and self.mode in set("wa"):
            if self.exists(path):
                raise OSError("ZipFS can only be open for reading or writing, not both")
            raise FileNotFoundError(path)
        if "r" in self.mode and "w" in mode:
            raise OSError("ZipFS can only be open for reading or writing, not both")
        out = self.zip.open(path, mode.strip("b"), force_zip64=self.force_zip_64)
        if "r" in mode:
            info = self.info(path)
            out.size = info["size"]
            out.name = info["name"]
        return out

    def find(self, path, maxdepth=None, withdirs=False, detail=False, **kwargs):
        if maxdepth is not None and maxdepth < 1:
            raise ValueError("maxdepth must be at least 1")

        # Remove the leading slash, as the zip file paths are always
        # given without a leading slash
        path = path.lstrip("/")
        path_parts = list(filter(lambda s: bool(s), path.split("/")))

        def _matching_starts(file_path):
            file_parts = filter(lambda s: bool(s), file_path.split("/"))
            return all(a == b for a, b in zip(path_parts, file_parts))

        self._get_dirs()

        result = {}
        # To match posix find, if an exact file name is given, we should
        # return only that file
        if path in self.dir_cache and self.dir_cache[path]["type"] == "file":
            result[path] = self.dir_cache[path]
            return result if detail else [path]

        for file_path, file_info in self.dir_cache.items():
            if not (path == "" or _matching_starts(file_path)):
                continue

            if file_info["type"] == "directory":
                if withdirs:
                    if file_path not in result:
                        result[file_path.strip("/")] = file_info
                continue

            if file_path not in result:
                result[file_path] = file_info if detail else None

        if maxdepth:
            path_depth = path.count("/")
            result = {
                k: v for k, v in result.items() if k.count("/") - path_depth < maxdepth
            }
        return result if detail else sorted(result)
