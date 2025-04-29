import copy
import dataclasses
import logging
import os
from enum import Enum
from typing import Optional, Union

from torch._inductor.remote_cache import JsonDataTy, RemoteCacheJsonSerde
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.utils._appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)


class CacheArtifactType(Enum):
    """
    Type of cache
    """

    INDUCTOR = 0
    AUTOTUNE = 1
    AOT_AUTOGRAD = 2
    PGO = 3


@dataclasses.dataclass(frozen=True)
class CacheArtifact:
    """
    Data for each cache artifact that will be serialized and deserialized
    """

    type: CacheArtifactType
    key: str
    content: bytes = dataclasses.field(repr=False)  # Do not display potential binary

    @staticmethod
    def serialize(writer: BytesWriter, cls: "CacheArtifact") -> None:
        writer.write_uint64(cls.type.value)
        writer.write_str(cls.key)
        writer.write_bytes(cls.content)

    @staticmethod
    def deserialize(reader: BytesReader) -> "CacheArtifact":
        type = reader.read_uint64()
        key = reader.read_str()
        content = reader.read_bytes()
        return CacheArtifact(CacheArtifactType(type), key, content)


@dataclasses.dataclass
class CacheInfo:
    """
    Return value of serialization and deserialization for the purpose of
    instrumentation
    """

    inductor_artifacts: list[str] = dataclasses.field(default_factory=list)
    autotune_artifacts: list[str] = dataclasses.field(default_factory=list)
    aot_autograd_artifacts: list[str] = dataclasses.field(default_factory=list)
    pgo_artifacts: list[str] = dataclasses.field(default_factory=list)

    def add(self, artifact: CacheArtifact) -> None:
        if artifact.type == CacheArtifactType.INDUCTOR:
            self.inductor_artifacts.append(artifact.key)
        elif artifact.type == CacheArtifactType.AUTOTUNE:
            self.autotune_artifacts.append(artifact.key)
        elif artifact.type == CacheArtifactType.AOT_AUTOGRAD:
            self.aot_autograd_artifacts.append(artifact.key)
        elif artifact.type == CacheArtifactType.PGO:
            self.pgo_artifacts.append(artifact.key)
        else:
            log.warning(f"Unsupported artifact type {artifact.type}")  # noqa: G004

    def clear(self) -> None:
        self.inductor_artifacts.clear()
        self.autotune_artifacts.clear()
        self.aot_autograd_artifacts.clear()
        self.pgo_artifacts.clear()


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
    _new_cache_artifacts: list[CacheArtifact] = []
    # Keep a seperate seen artifacts list to make avoid unnecessary duplicates
    # This list will not be cleared between serialize() calls
    _seen_artifacts: OrderedSet[CacheArtifact] = OrderedSet()
    # When serialize() is called, artifacts are transferred from _cache_artifacts to
    # internal data structure of the _serializer
    # This allows us to only pay the cost of serialization if serialize() is called
    _serializer: AppendingByteSerializer[CacheArtifact] = AppendingByteSerializer(
        serialize_fn=CacheArtifact.serialize
    )
    _cache_info: CacheInfo = CacheInfo()

    @classmethod
    def clear(cls) -> None:
        cls._new_cache_artifacts.clear()
        cls._seen_artifacts.clear()
        cls._serializer.clear()
        cls._cache_info.clear()

    @classmethod
    def record_artifact(
        cls,
        artifact_type: CacheArtifactType,
        key: str,
        content: Union[bytes, JsonDataTy],
    ) -> None:
        """
        Called from each caching operation to record the artifact in this
        "mega" list
        """
        if artifact_type == CacheArtifactType.AUTOTUNE:
            assert not isinstance(content, bytes)
            serde = RemoteCacheJsonSerde()
            content = serde.encode(content)
        assert isinstance(content, bytes)
        artifact = CacheArtifact(artifact_type, key, content)
        if artifact in cls._seen_artifacts:
            return
        log.debug("Recording %s", str(artifact))
        cls._new_cache_artifacts.append(artifact)
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
        for artifact in cls._new_cache_artifacts:
            log.debug("saving: %s", artifact)
            cls._cache_info.add(artifact)
        try:
            # We deep copy cls._cache_info since later compilations
            # can keep adding to cache_info
            info = copy.deepcopy(cls._cache_info)
            cls._serializer.extend(cls._new_cache_artifacts)
            artifact_bytes = cls._serializer.to_bytes()
            cls._new_cache_artifacts.clear()
            return artifact_bytes, info
        except Exception:
            log.warning("Failed to pickle cache artifacts", exc_info=True)
        return None

    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> Optional[CacheInfo]:
        """
        Converts the portable format back into various filesystem caches
        """
        try:
            artifacts = AppendingByteSerializer.to_list(
                serialized_artifacts, deserialize_fn=CacheArtifact.deserialize
            )
        except Exception:
            log.warning("Failed to un-pickle cache artifacts", exc_info=True)
            return None

        from torch._dynamo.pgo import write_local_impl
        from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
        from torch._inductor.codecache import FxGraphCache
        from torch._inductor.runtime.autotune_cache import _LocalAutotuneCacheBackend

        autotune_cache = _LocalAutotuneCacheBackend()

        info = CacheInfo()
        for artifact in artifacts:
            log.debug("writing: %s", artifact)
            info.add(artifact)

            if artifact.type == CacheArtifactType.INDUCTOR:
                FxGraphCache._write_to_local_cache(artifact.key, artifact.content)
            elif artifact.type == CacheArtifactType.AUTOTUNE:
                key = os.path.join(cache_dir(), artifact.key)
                autotune_cache._put(key, artifact.content)
            elif artifact.type == CacheArtifactType.AOT_AUTOGRAD:
                AOTAutogradCache._write_to_local_cache(artifact.key, artifact.content)
            elif artifact.type == CacheArtifactType.PGO:
                meta = write_local_impl(artifact.key, artifact.content)
                assert meta is not None
            else:
                log.warning(f"Unsupported artifact type {artifact.type}")  # noqa: G004
        return info
