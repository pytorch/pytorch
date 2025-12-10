"""Helper functions for a standard streaming compression API"""

from zipfile import ZipFile

import fsspec.utils
from fsspec.spec import AbstractBufferedFile


def noop_file(file, mode, **kwargs):
    return file


# TODO: files should also be available as contexts
# should be functions of the form func(infile, mode=, **kwargs) -> file-like
compr = {None: noop_file}


def register_compression(name, callback, extensions, force=False):
    """Register an "inferable" file compression type.

    Registers transparent file compression type for use with fsspec.open.
    Compression can be specified by name in open, or "infer"-ed for any files
    ending with the given extensions.

    Args:
        name: (str) The compression type name. Eg. "gzip".
        callback: A callable of form (infile, mode, **kwargs) -> file-like.
            Accepts an input file-like object, the target mode and kwargs.
            Returns a wrapped file-like object.
        extensions: (str, Iterable[str]) A file extension, or list of file
            extensions for which to infer this compression scheme. Eg. "gz".
        force: (bool) Force re-registration of compression type or extensions.

    Raises:
        ValueError: If name or extensions already registered, and not force.

    """
    if isinstance(extensions, str):
        extensions = [extensions]

    # Validate registration
    if name in compr and not force:
        raise ValueError(f"Duplicate compression registration: {name}")

    for ext in extensions:
        if ext in fsspec.utils.compressions and not force:
            raise ValueError(f"Duplicate compression file extension: {ext} ({name})")

    compr[name] = callback

    for ext in extensions:
        fsspec.utils.compressions[ext] = name


def unzip(infile, mode="rb", filename=None, **kwargs):
    if "r" not in mode:
        filename = filename or "file"
        z = ZipFile(infile, mode="w", **kwargs)
        fo = z.open(filename, mode="w")
        fo.close = lambda closer=fo.close: closer() or z.close()
        return fo
    z = ZipFile(infile)
    if filename is None:
        filename = z.namelist()[0]
    return z.open(filename, mode="r", **kwargs)


register_compression("zip", unzip, "zip")

try:
    from bz2 import BZ2File
except ImportError:
    pass
else:
    register_compression("bz2", BZ2File, "bz2")

try:  # pragma: no cover
    from isal import igzip

    def isal(infile, mode="rb", **kwargs):
        return igzip.IGzipFile(fileobj=infile, mode=mode, **kwargs)

    register_compression("gzip", isal, "gz")
except ImportError:
    from gzip import GzipFile

    register_compression(
        "gzip", lambda f, **kwargs: GzipFile(fileobj=f, **kwargs), "gz"
    )

try:
    from lzma import LZMAFile

    register_compression("lzma", LZMAFile, "lzma")
    register_compression("xz", LZMAFile, "xz")
except ImportError:
    pass

try:
    import lzmaffi

    register_compression("lzma", lzmaffi.LZMAFile, "lzma", force=True)
    register_compression("xz", lzmaffi.LZMAFile, "xz", force=True)
except ImportError:
    pass


class SnappyFile(AbstractBufferedFile):
    def __init__(self, infile, mode, **kwargs):
        import snappy

        super().__init__(
            fs=None, path="snappy", mode=mode.strip("b") + "b", size=999999999, **kwargs
        )
        self.infile = infile
        if "r" in mode:
            self.codec = snappy.StreamDecompressor()
        else:
            self.codec = snappy.StreamCompressor()

    def _upload_chunk(self, final=False):
        self.buffer.seek(0)
        out = self.codec.add_chunk(self.buffer.read())
        self.infile.write(out)
        return True

    def seek(self, loc, whence=0):
        raise NotImplementedError("SnappyFile is not seekable")

    def seekable(self):
        return False

    def _fetch_range(self, start, end):
        """Get the specified set of bytes from remote"""
        data = self.infile.read(end - start)
        return self.codec.decompress(data)


try:
    import snappy

    snappy.compress(b"")
    # Snappy may use the .sz file extension, but this is not part of the
    # standard implementation.
    register_compression("snappy", SnappyFile, [])

except (ImportError, NameError, AttributeError):
    pass

try:
    import lz4.frame

    register_compression("lz4", lz4.frame.open, "lz4")
except ImportError:
    pass

try:
    # zstd in the standard library for python >= 3.14
    from compression.zstd import ZstdFile

    register_compression("zstd", ZstdFile, "zst")

except ImportError:
    try:
        import zstandard as zstd

        def zstandard_file(infile, mode="rb"):
            if "r" in mode:
                cctx = zstd.ZstdDecompressor()
                return cctx.stream_reader(infile)
            else:
                cctx = zstd.ZstdCompressor(level=10)
                return cctx.stream_writer(infile)

        register_compression("zstd", zstandard_file, "zst")
    except ImportError:
        pass


def available_compressions():
    """Return a list of the implemented compressions."""
    return list(compr)
