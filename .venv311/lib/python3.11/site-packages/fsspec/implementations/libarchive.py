from contextlib import contextmanager
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_int,
    c_longlong,
    c_void_p,
    cast,
    create_string_buffer,
)

import libarchive
import libarchive.ffi as ffi

from fsspec import open_files
from fsspec.archive import AbstractArchiveFileSystem
from fsspec.implementations.memory import MemoryFile
from fsspec.utils import DEFAULT_BLOCK_SIZE

# Libarchive requires seekable files or memory only for certain archive
# types. However, since we read the directory first to cache the contents
# and also allow random access to any file, the file-like object needs
# to be seekable no matter what.

# Seek call-backs (not provided in the libarchive python wrapper)
SEEK_CALLBACK = CFUNCTYPE(c_longlong, c_int, c_void_p, c_longlong, c_int)
read_set_seek_callback = ffi.ffi(
    "read_set_seek_callback", [ffi.c_archive_p, SEEK_CALLBACK], c_int, ffi.check_int
)
new_api = hasattr(ffi, "NO_OPEN_CB")


@contextmanager
def custom_reader(file, format_name="all", filter_name="all", block_size=ffi.page_size):
    """Read an archive from a seekable file-like object.

    The `file` object must support the standard `readinto` and 'seek' methods.
    """
    buf = create_string_buffer(block_size)
    buf_p = cast(buf, c_void_p)

    def read_func(archive_p, context, ptrptr):
        # readinto the buffer, returns number of bytes read
        length = file.readinto(buf)
        # write the address of the buffer into the pointer
        ptrptr = cast(ptrptr, POINTER(c_void_p))
        ptrptr[0] = buf_p
        # tell libarchive how much data was written into the buffer
        return length

    def seek_func(archive_p, context, offset, whence):
        file.seek(offset, whence)
        # tell libarchvie the current position
        return file.tell()

    read_cb = ffi.READ_CALLBACK(read_func)
    seek_cb = SEEK_CALLBACK(seek_func)

    if new_api:
        open_cb = ffi.NO_OPEN_CB
        close_cb = ffi.NO_CLOSE_CB
    else:
        open_cb = libarchive.read.OPEN_CALLBACK(ffi.VOID_CB)
        close_cb = libarchive.read.CLOSE_CALLBACK(ffi.VOID_CB)

    with libarchive.read.new_archive_read(format_name, filter_name) as archive_p:
        read_set_seek_callback(archive_p, seek_cb)
        ffi.read_open(archive_p, None, open_cb, read_cb, close_cb)
        yield libarchive.read.ArchiveRead(archive_p)


class LibArchiveFileSystem(AbstractArchiveFileSystem):
    """Compressed archives as a file-system (read-only)

    Supports the following formats:
    tar, pax , cpio, ISO9660, zip, mtree, shar, ar, raw, xar, lha/lzh, rar
    Microsoft CAB, 7-Zip, WARC

    See the libarchive documentation for further restrictions.
    https://www.libarchive.org/

    Keeps file object open while instance lives. It only works in seekable
    file-like objects. In case the filesystem does not support this kind of
    file object, it is recommended to cache locally.

    This class is pickleable, but not necessarily thread-safe (depends on the
    platform). See libarchive documentation for details.
    """

    root_marker = ""
    protocol = "libarchive"
    cachable = False

    def __init__(
        self,
        fo="",
        mode="r",
        target_protocol=None,
        target_options=None,
        block_size=DEFAULT_BLOCK_SIZE,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fo: str or file-like
            Contains ZIP, and must exist. If a str, will fetch file using
            :meth:`~fsspec.open_files`, which must return one file exactly.
        mode: str
            Currently, only 'r' accepted
        target_protocol: str (optional)
            If ``fo`` is a string, this value can be used to override the
            FS protocol inferred from a URL
        target_options: dict (optional)
            Kwargs passed when instantiating the target FS, if ``fo`` is
            a string.
        """
        super().__init__(self, **kwargs)
        if mode != "r":
            raise ValueError("Only read from archive files accepted")
        if isinstance(fo, str):
            files = open_files(fo, protocol=target_protocol, **(target_options or {}))
            if len(files) != 1:
                raise ValueError(
                    f'Path "{fo}" did not resolve to exactly one file: "{files}"'
                )
            fo = files[0]
        self.of = fo
        self.fo = fo.__enter__()  # the whole instance is a context
        self.block_size = block_size
        self.dir_cache = None

    @contextmanager
    def _open_archive(self):
        self.fo.seek(0)
        with custom_reader(self.fo, block_size=self.block_size) as arc:
            yield arc

    @classmethod
    def _strip_protocol(cls, path):
        # file paths are always relative to the archive root
        return super()._strip_protocol(path).lstrip("/")

    def _get_dirs(self):
        fields = {
            "name": "pathname",
            "size": "size",
            "created": "ctime",
            "mode": "mode",
            "uid": "uid",
            "gid": "gid",
            "mtime": "mtime",
        }

        if self.dir_cache is not None:
            return

        self.dir_cache = {}
        list_names = []
        with self._open_archive() as arc:
            for entry in arc:
                if not entry.isdir and not entry.isfile:
                    # Skip symbolic links, fifo entries, etc.
                    continue
                self.dir_cache.update(
                    {
                        dirname: {"name": dirname, "size": 0, "type": "directory"}
                        for dirname in self._all_dirnames(set(entry.name))
                    }
                )
                f = {key: getattr(entry, fields[key]) for key in fields}
                f["type"] = "directory" if entry.isdir else "file"
                list_names.append(entry.name)

                self.dir_cache[f["name"]] = f
        # libarchive does not seem to return an entry for the directories (at least
        # not in all formats), so get the directories names from the files names
        self.dir_cache.update(
            {
                dirname: {"name": dirname, "size": 0, "type": "directory"}
                for dirname in self._all_dirnames(list_names)
            }
        )

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
        if mode != "rb":
            raise NotImplementedError

        data = b""
        with self._open_archive() as arc:
            for entry in arc:
                if entry.pathname != path:
                    continue

                if entry.size == 0:
                    # empty file, so there are no blocks
                    break

                for block in entry.get_blocks(entry.size):
                    data = block
                    break
                else:
                    raise ValueError
        return MemoryFile(fs=self, path=path, data=data)
