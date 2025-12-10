import base64
import io
from urllib.parse import unquote

from fsspec import AbstractFileSystem


class DataFileSystem(AbstractFileSystem):
    """A handy decoder for data-URLs

    Example
    -------
    >>> with fsspec.open("data:,Hello%2C%20World%21") as f:
    ...     print(f.read())
    b"Hello, World!"

    See https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs
    """

    protocol = "data"

    def __init__(self, **kwargs):
        """No parameters for this filesystem"""
        super().__init__(**kwargs)

    def cat_file(self, path, start=None, end=None, **kwargs):
        pref, data = path.split(",", 1)
        if pref.endswith("base64"):
            return base64.b64decode(data)[start:end]
        return unquote(data).encode()[start:end]

    def info(self, path, **kwargs):
        pref, name = path.split(",", 1)
        data = self.cat_file(path)
        mime = pref.split(":", 1)[1].split(";", 1)[0]
        return {"name": name, "size": len(data), "type": "file", "mimetype": mime}

    def _open(
        self,
        path,
        mode="rb",
        block_size=None,
        autocommit=True,
        cache_options=None,
        **kwargs,
    ):
        if "r" not in mode:
            raise ValueError("Read only filesystem")
        return io.BytesIO(self.cat_file(path))

    @staticmethod
    def encode(data: bytes, mime: str | None = None):
        """Format the given data into data-URL syntax

        This version always base64 encodes, even when the data is ascii/url-safe.
        """
        return f"data:{mime or ''};base64,{base64.b64encode(data).decode()}"
