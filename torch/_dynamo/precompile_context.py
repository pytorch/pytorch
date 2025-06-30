from abc import abstractmethod
from collections import defaultdict
from typing import Any, Generic, Optional, TypeVar
from typing_extensions import override

from torch.compiler._cache import (
    _serialize_single_cache,
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
    CacheArtifactsResult,
    CacheInfo,
)
from torch.utils._appending_byte_serializer import AppendingByteSerializer
from torch.utils._ordered_set import OrderedSet


"""
Classes and implementations related to precompile
"""

T = TypeVar("T")


class PrecompileCacheArtifact(CacheArtifact, Generic[T]):
    """
    Data for each cache artifact that will be serialized and deserialized by
    PrecompileContext, rather than CacheArtifactManager.
    T represents the deserialized type of the artifact, i.e. the return type of after_deserialization

    PrecompileCacheArtifact is a frozen dataclass - you can add new serializable fields and metadata specific to your own artifacts
    as needed, and use them in after_deserialization.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """

    @override
    def populate_cache(self) -> None:
        raise RuntimeError("Precompile cache artifacts do not populate caches")

    @override
    def precompile_compatible(self) -> bool:
        return True

    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...


class PrecompileContext(CacheArtifactManager):
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact
     - CodeStateArtifact (from torch._dynamo.package once available)
    """

    # Protected by the compile_lock
    # _new_cache_artifacts_by_key organizes results by the key of each artifact.
    # This allows us to implement serialize_by_key easily.
    # On call to `serialize()`, all cache artifacts in _new_cache_artifacts_by_key
    # are transferred to _new_cache_artifacts before serialization.
    _new_cache_artifacts_by_key: dict[str, CacheArtifact] = {}
    _new_cache_artifacts: CacheArtifactsResult = defaultdict(list)
    # Keep a separate seen artifacts list to make avoid unnecessary duplicates
    # This list will not be cleared between serialize() calls
    _seen_artifacts: OrderedSet[CacheArtifact] = OrderedSet()
    # When serialize() is called, artifacts are transferred from _cache_artifacts to
    # internal data structure of the _serializer
    # This allows us to only pay the cost of serialization if serialize() is called
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]] = (
        AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    )
    _cache_info: CacheInfo = CacheInfo()

    @classmethod
    def clear(cls) -> None:
        cls._new_cache_artifacts_by_key.clear()
        super().clear()

    @override
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
        # TODO: although this covers completely same artifacts, it's possible
        # with AOTAutogradCacheEntries to have multiple artifacts whose keys
        # (i.e. backend_ids) are different, but whose contents are equal.
        # In those cases, it would be much better if we only serialize once instead
        # of N times.
        if artifact in cls._seen_artifacts:
            return

        cls._new_cache_artifacts_by_key[key] = artifact
        cls._seen_artifacts.add(artifact)

    @classmethod
    def _save_artifacts_by_type(cls) -> None:
        """
        We normally record artifacts by key, but serialization expects them to be organized
        by artifact type. This function transfers artifacts from _new_cache_artifacts_by_key to _new_cache_artifacts
        """
        for artifact in cls._new_cache_artifacts_by_key.values():
            cls._new_cache_artifacts[artifact.__class__.type()].append(artifact)
        cls._new_cache_artifacts_by_key.clear()

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> Optional[CacheArtifact]:
        """
        Serialize all artifacts with the given key returned in a list.
        """
        return cls._new_cache_artifacts_by_key.get(key, None)

    @classmethod
    def serialize(cls) -> Optional[tuple[bytes, CacheInfo]]:
        cls._save_artifacts_by_type()
        return super().serialize()

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        raise NotImplementedError("TODO")

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            BundledAOTAutogradCacheArtifact,
        )
