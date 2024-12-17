import dataclasses
import logging
import os
import pickle
from enum import Enum
from typing import Optional, Union

from torch._inductor.remote_cache import JsonDataTy, RemoteCacheJsonSerde
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.utils._ordered_set import OrderedSet

from . import config


log = logging.getLogger(__name__)


class CacheArtifactType(Enum):
    """
    Type of cache
    """

    INDUCTOR = 0
    AUTOTUNE = 1
    AOT_AUTOGRAD = 2  # NYI
    PGO = 3  # NYI


@dataclasses.dataclass(frozen=True)
class CacheArtifact:
    """
    Cache artifact
    """

    type: CacheArtifactType
    key: str
    content: bytes = dataclasses.field(repr=False)  # Do not display potential binary


class CacheArtifactManager:
    """
    Lightweight manager class for collecting and processing cache artifacts for
    hot loading
    """

    # Protected by the compile_lock
    # Using set to avoid duplicates
    _cache_artifacts: OrderedSet[CacheArtifact] = OrderedSet()

    @staticmethod
    def is_enabled() -> bool:
        return config.record_cache_artifacts

    @classmethod
    def record_artifact(
        cls,
        artifact_type: CacheArtifactType,
        key: str,
        content: Union[bytes, JsonDataTy],
    ) -> None:
        if not CacheArtifactManager.is_enabled():
            return
        if artifact_type == CacheArtifactType.AUTOTUNE:
            assert not isinstance(content, bytes)
            serde = RemoteCacheJsonSerde()
            content = serde.encode(content)
        assert isinstance(content, bytes)
        cls._cache_artifacts.add(CacheArtifact(artifact_type, key, content))

    @classmethod
    def serialize(cls) -> Optional[bytes]:
        if log.isEnabledFor(logging.DEBUG):
            for artifact in cls._cache_artifacts:
                log.debug("saving: %s", artifact)
        try:
            return pickle.dumps(cls._cache_artifacts)
        except Exception:
            log.warning("Failed to pickle cache artifacts", exc_info=True)
        return None

    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> None:
        try:
            artifacts = pickle.loads(serialized_artifacts)
        except Exception:
            log.warning("Failed to un-pickle cache artifacts", exc_info=True)
            return

        from torch._inductor.codecache import FxGraphCache
        from torch._inductor.runtime.autotune_cache import _LocalAutotuneCacheBackend

        autotune_cache = _LocalAutotuneCacheBackend()

        for artifact in artifacts:
            log.debug("writing: %s", artifact)

            if artifact.type == CacheArtifactType.INDUCTOR:
                FxGraphCache._write_to_local_cache(artifact.key, artifact.content)
            elif artifact.type == CacheArtifactType.AUTOTUNE:
                key = os.path.join(cache_dir(), artifact.key)
                autotune_cache._put(key, artifact.content)
            else:
                log.warning(f"Unsupported artifact type {artifact.type}")  # noqa: G004
