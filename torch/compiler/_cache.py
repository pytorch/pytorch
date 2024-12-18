import dataclasses
import logging
import os
import pickle
from enum import Enum
from typing import List, Optional, Tuple, Union

from torch._inductor.remote_cache import JsonDataTy, RemoteCacheJsonSerde
from torch._inductor.runtime.runtime_utils import cache_dir

from . import config


log = logging.getLogger(__name__)


class CacheArtifactType(Enum):
    """
    Type of cache
    """

    INDUCTOR = 0
    AUTOTUNE = 1
    AOT_AUTOGRAD = 2  # NYI
    PGO = 3


@dataclasses.dataclass(frozen=True)
class CacheArtifact:
    """
    Data for each cache artifact that will be serialized and deserialized
    """

    type: CacheArtifactType
    key: str
    content: bytes = dataclasses.field(repr=False)  # Do not display potential binary


@dataclasses.dataclass
class CacheInfo:
    """
    Return value of serialization and deserialization for the purpose of
    instrumentation
    """

    inductor_artifacts: List[str] = dataclasses.field(default_factory=list)
    autotune_artifacts: List[str] = dataclasses.field(default_factory=list)
    aot_autograd_artifacts: List[str] = dataclasses.field(default_factory=list)
    pgo_artifacts: List[str] = dataclasses.field(default_factory=list)

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


class CacheArtifactManager:
    """
    Lightweight manager class for collecting and processing cache artifacts for
    hot loading

    Intended Lifecycle:
    - Set torch.compiler.config.record_cache_artifacts = True
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
    _cache_artifacts: List[CacheArtifact] = []

    @staticmethod
    def is_enabled() -> bool:
        return config.record_cache_artifacts

    @classmethod
    def clear(cls) -> None:
        cls._cache_artifacts.clear()

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
        if not CacheArtifactManager.is_enabled():
            return
        if artifact_type == CacheArtifactType.AUTOTUNE:
            assert not isinstance(content, bytes)
            serde = RemoteCacheJsonSerde()
            content = serde.encode(content)
        assert isinstance(content, bytes)
        cls._cache_artifacts.append(CacheArtifact(artifact_type, key, content))

    @classmethod
    def serialize(cls) -> Optional[Tuple[bytes, CacheInfo]]:
        """
        Converts the "mega" list into portable format
        """
        info = CacheInfo()
        for artifact in cls._cache_artifacts:
            log.debug("saving: %s", artifact)
            info.add(artifact)
        try:
            return (pickle.dumps(cls._cache_artifacts), info)
        except Exception:
            log.warning("Failed to pickle cache artifacts", exc_info=True)
        return None

    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> Optional[CacheInfo]:
        """
        Converst the portable format back into various filesystem caches
        """
        try:
            artifacts = pickle.loads(serialized_artifacts)
        except Exception:
            log.warning("Failed to un-pickle cache artifacts", exc_info=True)
            return None

        from torch._dynamo.pgo import write_local_impl
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
                raise AssertionError("not yet implemented")
            elif artifact.type == CacheArtifactType.PGO:
                meta = write_local_impl(artifact.key, artifact.content)
                assert meta is not None
            else:
                log.warning(f"Unsupported artifact type {artifact.type}")  # noqa: G004
        return info
