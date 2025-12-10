import errno
import io
import os
import secrets
import shutil
from contextlib import suppress
from functools import cached_property, wraps
from urllib.parse import parse_qs

from fsspec.spec import AbstractFileSystem
from fsspec.utils import (
    get_package_version_without_import,
    infer_storage_options,
    mirror_from,
    tokenize,
)


def wrap_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as exception:
            if not exception.args:
                raise

            message, *args = exception.args
            if isinstance(message, str) and "does not exist" in message:
                raise FileNotFoundError(errno.ENOENT, message) from exception
            else:
                raise

    return wrapper


PYARROW_VERSION = None


class ArrowFSWrapper(AbstractFileSystem):
    """FSSpec-compatible wrapper of pyarrow.fs.FileSystem.

    Parameters
    ----------
    fs : pyarrow.fs.FileSystem

    """

    root_marker = "/"

    def __init__(self, fs, **kwargs):
        global PYARROW_VERSION
        PYARROW_VERSION = get_package_version_without_import("pyarrow")
        self.fs = fs
        super().__init__(**kwargs)

    @property
    def protocol(self):
        return self.fs.type_name

    @cached_property
    def fsid(self):
        return "hdfs_" + tokenize(self.fs.host, self.fs.port)

    @classmethod
    def _strip_protocol(cls, path):
        ops = infer_storage_options(path)
        path = ops["path"]
        if path.startswith("//"):
            # special case for "hdfs://path" (without the triple slash)
            path = path[1:]
        return path

    def ls(self, path, detail=False, **kwargs):
        path = self._strip_protocol(path)
        from pyarrow.fs import FileSelector

        try:
            entries = [
                self._make_entry(entry)
                for entry in self.fs.get_file_info(FileSelector(path))
            ]
        except (FileNotFoundError, NotADirectoryError):
            entries = [self.info(path, **kwargs)]
        if detail:
            return entries
        else:
            return [entry["name"] for entry in entries]

    def info(self, path, **kwargs):
        path = self._strip_protocol(path)
        [info] = self.fs.get_file_info([path])
        return self._make_entry(info)

    def exists(self, path):
        path = self._strip_protocol(path)
        try:
            self.info(path)
        except FileNotFoundError:
            return False
        else:
            return True

    def _make_entry(self, info):
        from pyarrow.fs import FileType

        if info.type is FileType.Directory:
            kind = "directory"
        elif info.type is FileType.File:
            kind = "file"
        elif info.type is FileType.NotFound:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), info.path)
        else:
            kind = "other"

        return {
            "name": info.path,
            "size": info.size,
            "type": kind,
            "mtime": info.mtime,
        }

    @wrap_exceptions
    def cp_file(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip("/")
        path2 = self._strip_protocol(path2).rstrip("/")

        with self._open(path1, "rb") as lstream:
            tmp_fname = f"{path2}.tmp.{secrets.token_hex(6)}"
            try:
                with self.open(tmp_fname, "wb") as rstream:
                    shutil.copyfileobj(lstream, rstream)
                self.fs.move(tmp_fname, path2)
            except BaseException:
                with suppress(FileNotFoundError):
                    self.fs.delete_file(tmp_fname)
                raise

    @wrap_exceptions
    def mv(self, path1, path2, **kwargs):
        path1 = self._strip_protocol(path1).rstrip("/")
        path2 = self._strip_protocol(path2).rstrip("/")
        self.fs.move(path1, path2)

    @wrap_exceptions
    def rm_file(self, path):
        path = self._strip_protocol(path)
        self.fs.delete_file(path)

    @wrap_exceptions
    def rm(self, path, recursive=False, maxdepth=None):
        path = self._strip_protocol(path).rstrip("/")
        if self.isdir(path):
            if recursive:
                self.fs.delete_dir(path)
            else:
                raise ValueError("Can't delete directories without recursive=False")
        else:
            self.fs.delete_file(path)

    @wrap_exceptions
    def _open(self, path, mode="rb", block_size=None, seekable=True, **kwargs):
        if mode == "rb":
            if seekable:
                method = self.fs.open_input_file
            else:
                method = self.fs.open_input_stream
        elif mode == "wb":
            method = self.fs.open_output_stream
        elif mode == "ab":
            method = self.fs.open_append_stream
        else:
            raise ValueError(f"unsupported mode for Arrow filesystem: {mode!r}")

        _kwargs = {}
        if mode != "rb" or not seekable:
            if int(PYARROW_VERSION.split(".")[0]) >= 4:
                # disable compression auto-detection
                _kwargs["compression"] = None
        stream = method(path, **_kwargs)

        return ArrowFile(self, stream, path, mode, block_size, **kwargs)

    @wrap_exceptions
    def mkdir(self, path, create_parents=True, **kwargs):
        path = self._strip_protocol(path)
        if create_parents:
            self.makedirs(path, exist_ok=True)
        else:
            self.fs.create_dir(path, recursive=False)

    @wrap_exceptions
    def makedirs(self, path, exist_ok=False):
        path = self._strip_protocol(path)
        self.fs.create_dir(path, recursive=True)

    @wrap_exceptions
    def rmdir(self, path):
        path = self._strip_protocol(path)
        self.fs.delete_dir(path)

    @wrap_exceptions
    def modified(self, path):
        path = self._strip_protocol(path)
        return self.fs.get_file_info(path).mtime

    def cat_file(self, path, start=None, end=None, **kwargs):
        kwargs.setdefault("seekable", start not in [None, 0])
        return super().cat_file(path, start=None, end=None, **kwargs)

    def get_file(self, rpath, lpath, **kwargs):
        kwargs.setdefault("seekable", False)
        super().get_file(rpath, lpath, **kwargs)


@mirror_from(
    "stream",
    [
        "read",
        "seek",
        "tell",
        "write",
        "readable",
        "writable",
        "close",
        "seekable",
    ],
)
class ArrowFile(io.IOBase):
    def __init__(self, fs, stream, path, mode, block_size=None, **kwargs):
        self.path = path
        self.mode = mode

        self.fs = fs
        self.stream = stream

        self.blocksize = self.block_size = block_size
        self.kwargs = kwargs

    def __enter__(self):
        return self

    @property
    def size(self):
        return self.stream.size()

    def __exit__(self, *args):
        return self.close()


class HadoopFileSystem(ArrowFSWrapper):
    """A wrapper on top of the pyarrow.fs.HadoopFileSystem
    to connect it's interface with fsspec"""

    protocol = "hdfs"

    def __init__(
        self,
        host="default",
        port=0,
        user=None,
        kerb_ticket=None,
        replication=3,
        extra_conf=None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        host: str
            Hostname, IP or "default" to try to read from Hadoop config
        port: int
            Port to connect on, or default from Hadoop config if 0
        user: str or None
            If given, connect as this username
        kerb_ticket: str or None
            If given, use this ticket for authentication
        replication: int
            set replication factor of file for write operations. default value is 3.
        extra_conf: None or dict
            Passed on to HadoopFileSystem
        """
        from pyarrow.fs import HadoopFileSystem

        fs = HadoopFileSystem(
            host=host,
            port=port,
            user=user,
            kerb_ticket=kerb_ticket,
            replication=replication,
            extra_conf=extra_conf,
        )
        super().__init__(fs=fs, **kwargs)

    @staticmethod
    def _get_kwargs_from_urls(path):
        ops = infer_storage_options(path)
        out = {}
        if ops.get("host", None):
            out["host"] = ops["host"]
        if ops.get("username", None):
            out["user"] = ops["username"]
        if ops.get("port", None):
            out["port"] = ops["port"]
        if ops.get("url_query", None):
            queries = parse_qs(ops["url_query"])
            if queries.get("replication", None):
                out["replication"] = int(queries["replication"][0])
        return out
