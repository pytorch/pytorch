from fsspec.implementations.http import HTTPFileSystem

from ..base_io import BaseIO

class HTTPIO(BaseIO):
    def __init__(self, url) -> None:
        self.cur = 0
        self.url = url
        fs = HTTPFileSystem()
        self.f = fs.open(url, 'rb')

    def seek(self, offset: int, whence=0) -> int:
        return self.f.seek(offset, whence)

    def tell(self) -> int:
        return self.f.tell()

    def read(self, size: int = -1, /) -> bytes | None:
        return self.f.read(size)

    def readline(self, size: int | None = -1, /) -> bytes:
        if size < 0:
            line = self._readline(self.read)
        else:
            line = self.read(size)

        return line

    def close(self):
        self.f.close()
