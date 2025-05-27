from abc import abstractmethod
from collections import defaultdict
from typing import Any, Generic, Optional, TypeVar
from typing_extensions import override

from torch.compiler._cache import (
    _serialize_single_cache,
    CacheArtifact,
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
    It uses the same interface as CacheArtifactManager, but handles deserialization differently:
    specifically, it returns a full Callable object instead of populating caches. The goal of PrecompileContext
    is to provide a way for torch.compile to record and serialize various artifacts
    that are needed for precompile as it is compiling.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact

    In order to add new artifacts that are needed to return a callable from precompile,
    implement the class PrecompileCacheArtifact[T], where T is a serializable type.

    TODO: interface not yet finalized.
    """

    # Protected by the compile_lock
    _new_cache_artifacts: CacheArtifactsResult = defaultdict(list)
    # Keep a seperate seen artifacts list to make avoid unnecessary duplicates
    # This list will not be cleared between serialize() calls
    _seen_artifacts: OrderedSet[CacheArtifact] = OrderedSet()
    # When serialize() is called, artifacts are transferred from _cache_artifacts to
    # internal data structure of the _serializer
    # This allows us to only pay the cost of serialization if serialize() is called
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]] = (
        AppendingByteSerializer(serialize_fn=_serialize_single_cache)
    )
    _cache_info: CacheInfo = CacheInfo()

    @staticmethod
    # TODO: return a Callable here instead of the dict of artifacts
    def deserialize_into_callable(
        serialized_artifacts: bytes,
    ) -> Optional[dict[str, Any]]:
        """
        Similar to CacheArtifactManager.serialize, but instead of populating caches,
        we want to return a full Callable object.

        TODO: Add dynamo specific logic to convert dict[str, Any] into a Callable
        """
        artifacts: Optional[CacheArtifactsResult] = PrecompileContext.deserialize(
            serialized_artifacts
        )
        if not artifacts:
            return None
        results = {}
        for artifact_type, deserialized_artifacts in artifacts.items():
            result_artifacts = []
            for artifact in deserialized_artifacts:
                if not artifact.precompile_compatible():
                    raise RuntimeError(
                        f"Unsupported precompile artifact type: {artifact.__class__.type()}"
                    )
                assert isinstance(artifact, PrecompileCacheArtifact)
                result_artifacts.append(artifact.after_deserialization())
            results[artifact_type] = result_artifacts
        # TODO: return a Callable here instead of the dict of artifacts
        return results

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        raise RuntimeError("Precompile Contexts do not directly populate any caches")

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            BundledAOTAutogradCacheArtifact,
        )
