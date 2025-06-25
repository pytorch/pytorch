import base64
import zlib
from collections.abc import Iterable
from typing import Callable, Generic, TypeVar


T = TypeVar("T")

_ENCODING_VERSION: int = 1

__all__ = ["AppendingByteSerializer"]


#######################################
# Helper classes
#######################################

CHECKSUM_DIGEST_SIZE = 4


class BytesWriter:
    def __init__(self) -> None:
        # Reserve CHECKSUM_DIGEST_SIZE bytes for checksum
        self._data = bytearray(CHECKSUM_DIGEST_SIZE)

    def write_uint64(self, i: int) -> None:
        self._data.extend(i.to_bytes(8, byteorder="big", signed=False))

    def write_str(self, s: str) -> None:
        payload = base64.b64encode(s.encode("utf-8"))
        self.write_bytes(payload)

    def write_bytes(self, b: bytes) -> None:
        self.write_uint64(len(b))
        self._data.extend(b)

    def to_bytes(self) -> bytes:
        digest = zlib.crc32(self._data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        assert len(digest) == CHECKSUM_DIGEST_SIZE
        self._data[0:CHECKSUM_DIGEST_SIZE] = digest
        return bytes(self._data)


class BytesReader:
    def __init__(self, data: bytes) -> None:
        # Check for data corruption
        assert len(data) >= CHECKSUM_DIGEST_SIZE
        digest = zlib.crc32(data[CHECKSUM_DIGEST_SIZE:]).to_bytes(
            4, byteorder="big", signed=False
        )
        assert len(digest) == CHECKSUM_DIGEST_SIZE
        if data[0:CHECKSUM_DIGEST_SIZE] != digest:
            raise RuntimeError(
                "Bytes object is corrupted, checksum does not match. "
                f"Expected: {data[0:CHECKSUM_DIGEST_SIZE]!r}, Got: {digest!r}"
            )

        self._data = data
        self._i = CHECKSUM_DIGEST_SIZE

    def is_finished(self) -> bool:
        return len(self._data) == self._i

    def read_uint64(self) -> int:
        result = int.from_bytes(
            self._data[self._i : self._i + 8], byteorder="big", signed=False
        )
        self._i += 8
        return result

    def read_str(self) -> str:
        return base64.b64decode(self.read_bytes()).decode("utf-8")

    def read_bytes(self) -> bytes:
        size = self.read_uint64()
        result = self._data[self._i : self._i + size]
        self._i += size
        return result


#######################################
# AppendingByteSerializer
#######################################


class AppendingByteSerializer(Generic[T]):
    """
    Provides efficient serialization and deserialization of list of bytes
    Note that this does not provide any guarantees around byte order
    """

    _serialize_fn: Callable[[BytesWriter, T], None]
    _writer: BytesWriter

    def __init__(
        self,
        *,
        serialize_fn: Callable[[BytesWriter, T], None],
    ) -> None:
        self._serialize_fn = serialize_fn
        self.clear()

    def clear(self) -> None:
        self._writer = BytesWriter()
        # First 8-bytes are for version
        self._writer.write_uint64(_ENCODING_VERSION)

    def append(self, data: T) -> None:
        self._serialize_fn(self._writer, data)

    def extend(self, elems: Iterable[T]) -> None:
        for elem in elems:
            self.append(elem)

    def to_bytes(self) -> bytes:
        return self._writer.to_bytes()

    @staticmethod
    def to_list(data: bytes, *, deserialize_fn: Callable[[BytesReader], T]) -> list[T]:
        reader = BytesReader(data)
        assert reader.read_uint64() == _ENCODING_VERSION

        result: list[T] = []
        while not reader.is_finished():
            result.append(deserialize_fn(reader))
        return result
