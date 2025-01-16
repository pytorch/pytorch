# Copyright (c) Meta Platforms, Inc. and affiliates

import abc
import importlib.util
import sys
from collections.abc import Sequence
from typing import IO, Type


# NOTE: everything in this file is experimental, and subject to
# change.  Feedback and bug fixes are always welcome.


zstandard_module_name = "zstandard"

if (zstandard := sys.modules.get(zstandard_module_name, None)) is not None:
    pass
elif (zstandard_spec := importlib.util.find_spec(zstandard_module_name)) is not None:
    zstandard = importlib.util.module_from_spec(zstandard_spec)
    sys.modules[zstandard_module_name] = zstandard
    zstandard_spec.loader.exec_module(zstandard)  # type: ignore[union-attr]
else:
    zstandard = None


__all__ = [
    "Extension",
    "StreamTransformExtension",
    "ZStandard",
    "ExtensionRegistry",
]


class Extension(abc.ABC):
    """
    Extensions provide modular additions to functionality within distributed checkpointing,
    which affect the layout or format of the written artifacts.  Extensions may be
    built into pytorch, or provided externally.

    When writing, the caller provides a list of extension instances of the appropriate
    type.  Each extension can output a descriptor which is used to reconstitute the
    extension at read-time.
    """

    @staticmethod
    @abc.abstractmethod
    def registry_name() -> str:
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @staticmethod
    @abc.abstractmethod
    def from_descriptor(version: str) -> "Extension":
        """
        See ExtensionRegistry.from_descriptor_list
        """

    @abc.abstractmethod
    def get_descriptor(self) -> str:
        """
        Return descriptor name to be included in metadata.  The form should be
        "extension_name[@local-domain][/version]".
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
    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        """
        Takes a writeable output stream, and generates a new stream which implements the
        output transform.  Input data written to the returned stream will be transformed
        and written to the `output` argument stream.
        """

    @abc.abstractmethod
    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        """
        Takes a readable input stream, and generates a new stream which implements the
        input transform.  When the returned stream is read, data will be read from the
        'input' stream, transformed, and returned.
        """


class ZStandard(StreamTransformExtension):
    @staticmethod
    def is_available() -> bool:
        return zstandard is not None

    @staticmethod
    def from_descriptor(version: str) -> "ZStandard":
        if version.partition(".")[0] != "1":
            raise ValueError(f"Unknown extension {version=}")
        if not ZStandard.is_available():
            raise ValueError(
                f"Stream with ZStandard compression cannot be processed because no module named '{zstandard_module_name}'"
            )
        return ZStandard()

    @staticmethod
    def registry_name() -> str:
        return "stream.zstd"

    def __init__(self) -> None:
        super().__init__()
        if not ZStandard.is_available():
            raise ValueError(
                f"ZStandard extension is unavailable because no module named '{zstandard_module_name}'"
            )

    def get_descriptor(self) -> str:
        return f"{self.registry_name()}/1"

    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        compressor = zstandard.ZstdCompressor()  # type: ignore[union-attr]
        return compressor.stream_writer(output)

    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        decompressor = zstandard.ZstdDecompressor()  # type: ignore[union-attr]
        return decompressor.stream_reader(input)


class ExtensionRegistry:
    def __init__(self) -> None:
        # Populate default registry contents
        self.extensions: dict[str, Type[Extension]] = {
            cls.registry_name(): cls for cls in (ZStandard,)
        }

    def register(self, cls: Type[Extension]) -> None:
        self.extensions[cls.registry_name()] = cls

    def from_descriptor_list(self, descriptors: Sequence[str]) -> Sequence[Extension]:
        """
        Given a seuquence of descriptor strings as returned by
        Extension.get_descriptor at save time, creates a sequence of
        Extension instances.  The name[@local-domain] preceding the
        version number is used to look up an implementation class in
        the registry, and the version is passed to the class's
        from_descriptor static method.  If the registry contains no
        match, this will throw ValueError.  If the from_descriptor
        method raises an exception, that will pass through to the
        caller.
        """

        def from_descriptor(desc: str) -> Extension:
            name, _, version = desc.partition("/")
            if version is None:
                version = 0
            ext = self.extensions.get(name)
            if not ext:
                raise ValueError(f"Unknown extension {name=}")
            return ext.from_descriptor(version)

        return [from_descriptor(desc) for desc in descriptors]
