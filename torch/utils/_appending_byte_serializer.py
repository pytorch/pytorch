from collections.abc import Iterable
from typing import Callable, Generic, TypeVar


T = TypeVar("T")

_ENCODING_VERSION: int = 1

__all__ = ["AppendingByteSerializer"]


#######################################
# Helper classes
#######################################


class BytesWriter:
    def __init__(self) -> None:
        self._data = bytearray()

    def write_int(self, i: int) -> None:
        # Note: We are not doing anything about byteorder (big or little endian)
        self._data.extend(i.to_bytes(8, signed=False))

    def write_str(self, s: str) -> None:
        payload = s.encode("utf-8")
        self.write_int(len(payload))
        self.write_bytes(payload)

    def write_bytes(self, b: bytes) -> None:
        self._data.extend(b)

    def get(self) -> bytes:
        return bytes(self._data)


class BytesReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._i = 0

    def is_finished(self) -> bool:
        return len(self._data) == self._i

    def read_int(self) -> int:
        # Note: We are not doing anything about byteorder (big or little endian)
        result = int.from_bytes(self._data[self._i : self._i + 8], signed=False)
        self._i += 8
        return result

    def read_str(self) -> str:
        size = self.read_int()
        result = self._data[self._i : self._i + size].decode("utf-8")
        self._i += size
        return result

    def read_bytes(self, fixed_len: int) -> bytes:
        result = self._data[self._i : self._i + fixed_len]
        self._i += fixed_len
        return result


#######################################
# AppendingByteSerializer
#######################################


class AppendingByteSerializer(Generic[T]):
    """
    Provides efficient serialization and deserialization of list of bytes
    Note that this does not provide any guarantees around byte order
    """

    _serialize_fn: Callable[[T], bytes]
    _writer: BytesWriter

    def __init__(self, *, serialize_fn: Callable[[T], bytes]) -> None:
        self._serialize_fn = serialize_fn
        self.clear()

    def clear(self) -> None:
        # Use first byte as version
        self._writer = BytesWriter()
        self._writer.write_int(_ENCODING_VERSION)

    def append(self, data: T) -> None:
        payload = self._serialize_fn(data)
        self._writer.write_int(len(payload))
        self._writer.write_bytes(payload)

    def get(self) -> bytes:
        return self._writer.get()

    def appendListAndGet(self, elems: Iterable[T]) -> bytes:
        for elem in elems:
            self.append(elem)
        return self.get()

    @staticmethod
    def to_list(data: bytes, *, deserialize_fn: Callable[[bytes], T]) -> list[T]:
        reader = BytesReader(data)
        assert reader.read_int() == _ENCODING_VERSION

        result: list[T] = []
        while not reader.is_finished():
            payload_len = reader.read_int()
            result.append(deserialize_fn(reader.read_bytes(payload_len)))
        return result
