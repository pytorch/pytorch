# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
import io
import os
import zstandard
from typing import Dict, Sequence, Type


__all__ = ["Extension", "StreamTransformExtension", "Rot13", "ZStandard", "ExtensionRegistry"]


class Extension(abc.ABC):
    """
    Extensions provide modular additions to functionality within distributed checkpointing,
    which affect the layour or format of the written artifacts.  Extensions may be
    built into pytorch, or provided externally.

    When writing, the caller provides a list of extension instances of the appropriate
    type.  Each extension can output a descriptor which is used to reconstitute the
    extension at read-time.
    """

    @abc.abstractmethod
    def get_descriptor() -> str:
        """
        Return descriptor name to be included in metadata
        """


class StreamTransformExtension(Extension):
    """
    An extension which performs transformation on a byte stream, such as compression
    or encryption.

    Implementations should try to be memory friendly and performant.  For example, don't
    read the whole input, then transform it, and write it back.  If at all possible, do it in
    chunks.  But, don't read/transform/write one byte at a time, either.
    """

    @abc.abstractmethod
    def transform_to(self, output: io.IOBase) -> io.RawIOBase:
        """
        Takes a writeable output stream, and generates a new stream which implements the
        output transform.  Input data written to the returned stream will be transformed
        and written to the `output` argument stream.
        """

    @abc.abstractmethod
    def transform_from(self, input: io.IOBase) -> io.RawIOBase:
        """
        Takes a readable input stream, and generates a new stream which implements the
        input transform.  When the returned stream is read, data will be read from the
        'input' stream, transformed, and returned.
        """


class Rot13(StreamTransformExtension):
    def __init__(self, chunk_size: int = io.DEFAULT_BUFFER_SIZE):
        super().__init__()
        self._chunk_size = chunk_size

    @staticmethod
    def from_descriptor(version: str) -> "Rot13":
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        return Rot13()

    @staticmethod
    def registry_name() -> str:
        return "stream.rot13"

    def get_descriptor(self):
        return f"{self.registry_name()}/1"

    @staticmethod
    def _rot13bytes(b, count):
        for i in range(count):
            ch = b[i]
            if ch >= ord('A') and ch <= ord('Z'):
                ch += ord('a') - ord('A')
            elif ch >= ord('a') and ch <= ord('z'):
                ch += ord('A') - ord('a')
            b[i] = ch

    def transform_to(self, output: io.IOBase) -> io.RawIOBase:
        class Writer(io.RawIOBase):
            def __init__(self, output: io.IOBase):
                self.output = output

            def writeable(self):
                return True

            def write(self, b):
                # Don't mutate the input
                chunk = bytearray(b)
                Rot13._rot13bytes(chunk, len(chunk))
                return self.output.write(chunk)

            def flush(self):
                self.output.flush()

        return Writer(output)

    def transform_from(self, input: io.IOBase) -> io.RawIOBase:
        class Reader(io.RawIOBase):
            def __init__(self, input: io.IOBase):
                self.input = input

            def readable(self):
                return True

            def readinto(self, b):
                count = self.input.readinto(b)
                if count == 0 or count is None:
                    return count

                Rot13._rot13bytes(b, count)
                return count

            def seekable(self):
                return self.input.seekable()

            def seek(self, offset, whence=os.SEEK_SET):
                return self.input.seek(offset, whence)

            def tell(self):
                return self.input.tell()

        return Reader(input)


class ZStandard(StreamTransformExtension):
    def __init__(self):
        super().__init__()

    @staticmethod
    def from_descriptor(version: str) -> "ZStandard":
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        return ZStandard()

    @staticmethod
    def registry_name() -> str:
        return "stream.zstd"

    def get_descriptor(self):
        return f"{self.registry_name()}/1"

    def transform_to(self, output: io.IOBase) -> io.RawIOBase:
        compressor = zstandard.ZstdCompressor()
        return compressor.stream_writer(output)

    def transform_from(self, input: io.IOBase) -> io.RawIOBase:
        decompressor = zstandard.ZstdDecompressor()
        return decompressor.stream_reader(input)

class ExtensionRegistry:
    def __init__(self):
        # Populate default registry contents
        self.extensions: Dict[Type[Extension]] = {
            cls.registry_name(): cls for cls in [Rot13, ZStandard]
        }

    def register(self, cls: Type[Extension]) -> None:
        self.extensions[cls.registry_name] = cls

    def from_descriptor_list(self, descriptors: Sequence[str]) -> Sequence[Extension]:
        def from_descriptor(desc: str) -> Extension:
            name, _, version = desc.partition("/")
            if version is None:
                version = 0
            ext = self.extensions.get(name)
            if not ext:
                raise ValueError(f"Unknown extension {name=}")
            return ext.from_descriptor(version)

        return [from_descriptor(desc) for desc in descriptors]
