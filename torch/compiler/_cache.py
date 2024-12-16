import dataclasses
import logging
import pickle
from enum import Enum
from typing import Optional

from torch.utils._ordered_set import OrderedSet

from . import config


log = logging.getLogger(__name__)


class CacheArtifactType(Enum):
    """
    Type of cache
    """

    INDUCTOR_CACHE = 0


@dataclasses.dataclass(frozen=True)
class CacheArtifact:
    """
    Cache artifact
    """

    type: CacheArtifactType
    key: str
    content: bytes


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
        cls, artifact_type: CacheArtifactType, key: str, content: bytes
    ) -> None:
        if not CacheArtifactManager.is_enabled():
            return
        cls._cache_artifacts.add(CacheArtifact(artifact_type, key, content))

    @classmethod
    def serialize(cls) -> Optional[bytes]:
        if not CacheArtifactManager.is_enabled():
            return None
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
        for artifact in artifacts:
            if artifact.type == CacheArtifactType.INDUCTOR_CACHE:
                pass
            else:
                log.warning(f"Unsupported artifact type {artifact.type}")  # noqa: G004
