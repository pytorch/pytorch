from urllib import parse
from pyarrow import fs

from ..base_io import BaseIO


class HDFSIO(BaseIO):
    def __init__(self, url: str, mode) -> None:
        parse_result = parse.urlparse(url, scheme='hdfs')
        self.path = parse_result.path
        # no path in hdfs uri
        self.uri = parse_result._replace(path='').geturl()
        self.hdfs = fs.HadoopFileSystem.from_uri(self.uri)

        self.reader = None
        self.writer = None
        if "w" in mode:
            self.writer = self.hdfs.open_output_stream(self.path)
        elif "r" in mode:
            self.reader = self.hdfs.open_input_file(self.path)
        else:
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")

    def seek(self, offset: int, whence=0) -> int:
        return self.reader.seek(offset, whence)

    def tell(self) -> int:
        return self.reader.tell()

    def read(self, size: int = -1, /) -> bytes | None:
        return self.reader.read(size)

    def readline(self, size: int | None = -1, /) -> bytes:
        if size < 0:
            line = self._readline(self.read)
        else:
            line = self.read(size)

        return line

    def write(self, b, /) -> int | None:
        return self.writer.write(b)

    def flush(self) -> None:
        return self.writer.flush()

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()

        if self.writer is not None:
            self.writer.close()
