import copy
import dataclasses
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from itertools import chain
from typing import Any, Optional

from torch.utils._appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class CacheArtifact(ABC):
    """
    Data for each cache artifact that will be serialized and deserialized
    """

    key: str
    content: bytes = dataclasses.field(repr=False)  # Do not display potential binary

    @staticmethod
    def serialize(writer: BytesWriter, cls: "CacheArtifact") -> None:
        writer.write_str(cls.key)
        writer.write_bytes(cls.content)

    @staticmethod
    def deserialize(artifact_type: str, reader: BytesReader) -> "CacheArtifact":
        key = reader.read_str()
        content = reader.read_bytes()
        return CacheArtifactFactory.create(artifact_type, key, content)

    @staticmethod
    def encode(content: Any) -> bytes:
        assert isinstance(content, bytes), f"Expected bytes, got {type(content)}"
        return content

    @abstractmethod
    def populate_cache(self) -> None:
        pass

    def precompile_compatible(self) -> bool:
        return False

    @staticmethod
    def type() -> str:
        """
        Returns the type of the artifact. Must be unique across all CacheArtifact classes.

        CacheArtifactFactory.register will add property method to CacheInfo based on this (def {type}_artifacts)
        that returns all artifacts for specific cache.
        """
        raise RuntimeError("CacheArtifact is an abstract class, please use a subclass")


class CacheArtifactFactory:
    """
    Factory for creating CacheArtifact objects based on their type
    """

    _artifact_types: dict[str, type[CacheArtifact]] = {}

    @classmethod
    def register(cls, artifact_cls: type[CacheArtifact]) -> type[CacheArtifact]:
        artifact_type_key = artifact_cls.type()
        assert (
            artifact_cls.type() not in cls._artifact_types
        ), f"Artifact of type={artifact_type_key} already registered in mega-cache artifact factory"
        cls._artifact_types[artifact_type_key] = artifact_cls
        setattr(
            CacheInfo,
            f"{artifact_type_key}_artifacts",
            property(lambda self: self.artifacts[artifact_type_key]),
        )
        return artifact_cls

    @classmethod
    def _get_artifact_type(cls, artifact_type_key: str) -> type[CacheArtifact]:
        assert (
            artifact_type_key in cls._artifact_types
        ), f"Artifact of type={artifact_type_key} not registered in mega-cache artifact factory"
        return cls._artifact_types[artifact_type_key]

    @classmethod
    def create(cls, artifact_type_key: str, key: str, content: bytes) -> CacheArtifact:
        artifact_cls = cls._get_artifact_type(artifact_type_key)
        return artifact_cls(key, content)

    @classmethod
    def encode_create(
        cls, artifact_type_key: str, key: str, content: Any
    ) -> CacheArtifact:
        artifact_cls = cls._get_artifact_type(artifact_type_key)
        return artifact_cls(key, artifact_cls.encode(content))


@dataclasses.dataclass
class CacheInfo:
    """
    Return value of serialization and deserialization for the purpose of
    instrumentation
    """

    artifacts: defaultdict[str, list[str]] = dataclasses.field(
        default_factory=lambda: defaultdict(list)
    )

    # Methods set by CacheArtifactFactory.register based on CacheArtifact.type()
    @property
    def inductor_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def autotune_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def aot_autograd_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def pgo_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    @property
    def precompile_aot_autograd_artifacts(self) -> list[str]:  # type: ignore[empty-body]
        ...

    def add(self, artifact: CacheArtifact) -> None:
        self.artifacts[artifact.type()].append(artifact.key)

    def clear(self) -> None:
        self.artifacts.clear()

    def empty(self) -> bool:
        return not self.artifacts


def _serialize_single_cache(
    writer: BytesWriter, cls: "tuple[str, list[CacheArtifact]]"
) -> None:
    writer.write_str(cls[0])
    writer.write_uint64(len(cls[1]))
    for artifact in cls[1]:
        CacheArtifact.serialize(writer, artifact)


def _deserialize_single_cache(
    reader: BytesReader,
) -> "tuple[str, list[CacheArtifact]]":
    artifacts = []
    artifact_type_key = reader.read_str()
    num_artifacts = reader.read_uint64()
    for _ in range(num_artifacts):
        artifacts.append(CacheArtifact.deserialize(artifact_type_key, reader))

    return artifact_type_key, artifacts


CacheArtifactsResult = dict[str, list[CacheArtifact]]


class CacheArtifactManager:
    """
    Lightweight manager class for collecting and processing cache artifacts for
    hot loading

    Intended Lifecycle:
    - Execute code via torch.compile, this will call
        CacheArtifactManager.record_artifact on each cache artifact
    - Call CacheArtifactManager.serialize to convert all the cache artifacts
        to portable format
    - Call CacheArtifactManager.deserialize to hot load the cache artifacts on
        a potentially different process

    NOTE: There's no FB/FC guarentees, results of cache artifacts will not be
          used unless code version matches.
    """

    # Protected by the compile_lock
    _new_cache_artifacts: CacheArtifactsResult = defaultdict(list)
    # Keep a seperate seen artifacts list to make avoid unnecessary duplicates
    # This list will not be cleared between serialize() calls
    _seen_artifacts: OrderedSet[CacheArtifact] = OrderedSet()
    # When serialize() is called, artifacts are transferred from _cache_artifacts to
    # internal data structure of the _serializer
    # This allows us to only pay the cost of serialization if serialize() is called
    _serializer: AppendingByteSerializer[
        tuple[str, list[CacheArtifact]]
    ] = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    _cache_info: CacheInfo = CacheInfo()

    @classmethod
    def clear(cls) -> None:
        cls._new_cache_artifacts.clear()
        cls._seen_artifacts.clear()
        cls._serializer.clear()
        cls._cache_info.clear()

    @classmethod
    @contextmanager
    def with_fresh_cache(cls) -> Generator[None, None, None]:
        original_new_cache_artifacts = cls._new_cache_artifacts
        original_seen_artifacts = cls._seen_artifacts
        original_serializer = cls._serializer
        original_cache_info = cls._cache_info

        cls._new_cache_artifacts = defaultdict(list)
        cls._seen_artifacts = OrderedSet()
        cls._serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
        cls._cache_info = cls._cache_info.__class__()
        try:
            yield
        finally:
            cls._new_cache_artifacts = original_new_cache_artifacts
            cls._seen_artifacts = original_seen_artifacts
            cls._serializer = original_serializer
            cls._cache_info = original_cache_info

    @classmethod
    def record_artifact(
        cls,
        artifact_type: str,
        key: str,
        content: Any,
    ) -> None:
        """
        Called from each caching operation to record the artifact in this
        "mega" list
        """
        artifact = CacheArtifactFactory.encode_create(artifact_type, key, content)
        if artifact in cls._seen_artifacts:
            return
        log.debug("Recording %s", str(artifact))
        cls._new_cache_artifacts[artifact_type].append(artifact)
        cls._seen_artifacts.add(artifact)

    @classmethod
    def need_serialize(cls) -> bool:
        """
        Have we seen new artifacts since last serialize call?
        """
        return len(cls._new_cache_artifacts) != 0

    @classmethod
    def serialize(cls) -> Optional[tuple[bytes, CacheInfo]]:
        """
        Converts the "mega" list into portable format
        """
        for artifact in chain(*cls._new_cache_artifacts.values()):
            log.debug("saving: %s", artifact)
            cls._cache_info.add(artifact)

        if cls._cache_info.empty():
            # If there are not artifacts, dont just return bytes with
            # version.
            return None

        try:
            # We deep copy cls._cache_info since later compilations
            # can keep adding to cache_info
            info = copy.deepcopy(cls._cache_info)
            cls._serializer.extend(cls._new_cache_artifacts.items())
            artifact_bytes = cls._serializer.to_bytes()
            cls._new_cache_artifacts.clear()
            return artifact_bytes, info
        except Exception:
            log.warning("Failed to pickle cache artifacts", exc_info=True)
        return None

    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> Optional[CacheArtifactsResult]:
        """
        Converts the portable format back into CacheArtifacts
        """
        try:
            CacheArtifactManager._ensure_cache_artifacts_registered()
            artifacts = dict(
                AppendingByteSerializer.to_list(
                    serialized_artifacts,
                    deserialize_fn=_deserialize_single_cache,
                )
            )
        except Exception:
            log.warning("Failed to un-pickle cache artifacts", exc_info=True)
            return None

        return artifacts

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        info = CacheInfo()
        for artifact in chain(*artifacts.values()):
            log.debug("writing: %s", artifact)
            info.add(artifact)
            artifact.populate_cache()

        return info

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        """When deserializing caches in fresh process, we need to ensure that all
        cache artifacts are registered in the cache registry. This is done by
        simply importing all the cache artifacts already wrapped with register call.
        """
        from torch._dynamo.pgo import PGOCacheArtifact  # noqa: F401
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            AOTAutogradCacheArtifact,
        )
        from torch._inductor.codecache import InductorCacheArtifact  # noqa: F401
        from torch._inductor.runtime.autotune_cache import (  # noqa: F401
            AutotuneCacheArtifact,
        )
