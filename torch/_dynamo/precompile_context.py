import copy
import pickle
from abc import abstractmethod
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Generic, Optional, TypeVar, Union
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


class EditablePrecompileCacheArtifact(Generic[T]):
    """
    A PrecompileCacheArtifact whose content isn't encoded until we call PrecompileContext.serialize()
    """

    def __init__(self, artifact_type: str, content: Any, key: str) -> None:
        # Deepcopy the content for now, but don't pickle it yet.
        # This allows us to make changes to self.content before true serialization
        self.content = copy.deepcopy(content)
        self.key = key
        self.artifact_type = artifact_type

    def real_encode(self) -> PrecompileCacheArtifact[T]:
        """
        Actually encode the object
        """
        content = pickle.dumps(self.content)
        artifact = CacheArtifactFactory.encode_create(
            self.artifact_type, self.key, content
        )
        assert isinstance(artifact, PrecompileCacheArtifact)
        return artifact

    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        self.content = edit_fn(self.content)


class PrecompileContext(CacheArtifactManager):
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact
     - DynamoCodeStateArtifact
     - AutotuneCacheArtifact (regular autotune results, same as Megacache)
    """

    # Protected by the compile_lock
    # _new_cache_artifacts_by_key organizes results by the key of each artifact.
    # This allows us to implement serialize_by_key easily.
    # On call to `serialize()`, all cache artifacts in _new_cache_artifacts_by_key
    # are transferred to _new_cache_artifacts before serialization.
    _new_cache_artifacts_by_key: dict[
        str, Union[EditablePrecompileCacheArtifact[object], CacheArtifact]
    ] = {}
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
        editable: bool = False,
    ) -> None:
        """
        Called from each caching operation to record the artifact in this
        "mega" list
        """
        artifact: Union[EditablePrecompileCacheArtifact[object], CacheArtifact]
        if editable:
            artifact = EditablePrecompileCacheArtifact(artifact_type, content, key)
        else:
            artifact = CacheArtifactFactory.encode_create(artifact_type, key, content)
            # TODO: although this covers completely same artifacts, it's possible
            # with AOTAutogradCacheEntries to have multiple artifacts whose keys
            # (i.e. backend_ids) are different, but whose contents are equal.
            # In those cases, it would be much better if we only serialize once instead
            # of N times.
            if artifact in cls._seen_artifacts:
                return
            cls._seen_artifacts.add(artifact)

        cls._new_cache_artifacts_by_key[key] = artifact

    @classmethod
    def _save_artifacts_by_type(cls) -> None:
        """
        We normally record artifacts by key, but serialization expects them to be organized
        by artifact type. This function transfers artifacts from _new_cache_artifacts_by_key to _new_cache_artifacts
        """
        for artifact in cls._new_cache_artifacts_by_key.values():
            if isinstance(artifact, EditablePrecompileCacheArtifact):
                artifact = artifact.real_encode()
            cls._new_cache_artifacts[artifact.__class__.type()].append(artifact)
        cls._new_cache_artifacts_by_key.clear()

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        assert key in cls._new_cache_artifacts_by_key, (
            f"Key {key} not found in artifacts"
        )
        artifact = cls._new_cache_artifacts_by_key[key]
        assert isinstance(artifact, EditablePrecompileCacheArtifact), (
            "Artifact is not editable"
        )
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> Optional[CacheArtifact]:
        """
        Serialize all artifacts with the given key returned in a list.
        """
        result = cls._new_cache_artifacts_by_key.get(key, None)
        if isinstance(result, EditablePrecompileCacheArtifact):
            result = result.real_encode()
        return result

    @classmethod
    def serialize(cls) -> Optional[tuple[bytes, CacheInfo]]:
        cls._save_artifacts_by_type()
        # No need to serialize if there are no new dynamo compiles
        if "precompile_dynamo" not in cls._new_cache_artifacts:
            return None
        return super().serialize()

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        PrecompileContext._ensure_cache_artifacts_registered()

        artifacts_by_key = {}
        cache_info = CacheInfo()
        for artifact in chain(*artifacts.values()):
            if artifact.type() == "autotune":
                # Populate autotune cache artifacts
                artifact.populate_cache()
            else:
                artifacts_by_key[artifact.key] = artifact
            cache_info.add(artifact)

        from torch._dynamo.package import _BackendId, DynamoCache

        for dynamo_entry in artifacts["precompile_dynamo"]:
            assert isinstance(dynamo_entry, PrecompileCacheArtifact)
            cache_entry = dynamo_entry.after_deserialization()
            # Grab backends from the dynamo cache entry
            backends = cache_entry.backend_ids
            backend_content: dict[_BackendId, PrecompileCacheArtifact[Any]] = {}
            for id_ in backends:
                assert id_ in artifacts_by_key, f"Backend {id_} not found in artifacts"
                artifact = artifacts_by_key[id_]
                assert isinstance(artifact, PrecompileCacheArtifact)
                backend_content[id_] = artifact
            DynamoCache.write(cache_entry, backend_content, dynamo_entry.key)

        return cache_info

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        from torch._dynamo.package import _DynamoCacheArtifact  # noqa: F401
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            BundledAOTAutogradCacheArtifact,
        )
