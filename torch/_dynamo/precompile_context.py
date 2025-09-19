import copy
import json
import logging
import pickle
from abc import abstractmethod
from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Generic, Optional, TypeVar, Union
from typing_extensions import override

import torch
from torch._dynamo.package import _DynamoCacheEntry
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
logger = logging.getLogger(__name__)


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


@CacheArtifactFactory.register
class EagerCacheArtifact(PrecompileCacheArtifact[Any]):
    @staticmethod
    def type() -> str:
        return "precompile_eager"

    def after_deserialization(self) -> Any:
        return pickle.loads(self.content)


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


@CacheArtifactFactory.register
class _DynamoCacheArtifact(PrecompileCacheArtifact[_DynamoCacheEntry]):
    @staticmethod
    def type() -> str:
        return "precompile_dynamo"

    def after_deserialization(self) -> _DynamoCacheEntry:
        result = pickle.loads(self.content)
        return result


class BypassDynamoCacheEntry(Exception):
    pass


class PrecompileContext(CacheArtifactManager):
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    PrecompileContext has two main portions: dynamo_cache_entries and backend_cache_artifacts.
    When saving, PrecompileContext.serialize() will serialize all dynamo cache entries along with any PrecompileCacheArtifacts that
    are needed to save those dynamo cache entries.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact
     - AutotuneCacheArtifact (regular autotune results, same as Megacache)

    """

    # Protected by the compile_lock
    # _backend_artifacts_by_key organizes results by the key of each artifact.
    # This allows us to implement serialize_by_key easily.
    # On call to `serialize()`, all cache artifacts in _backend_artifacts_by_key
    # are transferred to _new_cache_artifacts before serialization.
    _backend_artifacts_by_key: dict[
        str, Union[EditablePrecompileCacheArtifact[object], CacheArtifact]
    ] = {}

    # On call to `serialize()`, all cache artifacts in _dynamo_cache_entries are converted
    # into DynamoCacheArtifacts and added to _new_cache_artifacts for serialization
    _dynamo_cache_entries: dict[str, _DynamoCacheEntry] = {}

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
        cls._backend_artifacts_by_key.clear()
        cls._dynamo_cache_entries.clear()
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

        cls._backend_artifacts_by_key[key] = artifact

    @classmethod
    def record_dynamo_cache_entry(
        cls, cache_entry: _DynamoCacheEntry, key: str
    ) -> None:
        cls._dynamo_cache_entries[key] = cache_entry

    @classmethod
    def _save_artifacts_by_type(cls) -> None:
        """
        We normally record artifacts by key, but serialization expects them to be organized
        by artifact type. This function transfers artifacts from _backend_artifacts_by_key to _new_cache_artifacts
        """
        for key, cache_entry in cls._dynamo_cache_entries.items():
            backends = cache_entry.backend_ids
            try:
                for id_ in backends:
                    if id_ not in cls._backend_artifacts_by_key:
                        logger.warning(
                            "Bypassing %s because backend %s not found in artifacts"
                        )
                        raise BypassDynamoCacheEntry
            except BypassDynamoCacheEntry:
                continue
            pickled_result = pickle.dumps(cache_entry)
            dynamo_artifact = _DynamoCacheArtifact(key, pickled_result)
            cls._new_cache_artifacts[_DynamoCacheArtifact.type()].append(
                dynamo_artifact
            )

        # Save all the backend artifacts
        for artifact in cls._backend_artifacts_by_key.values():
            if isinstance(artifact, EditablePrecompileCacheArtifact):
                artifact = artifact.real_encode()
            cls._new_cache_artifacts[artifact.__class__.type()].append(artifact)
        cls._backend_artifacts_by_key.clear()

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        assert key in cls._backend_artifacts_by_key, f"Key {key} not found in artifacts"
        artifact = cls._backend_artifacts_by_key[key]
        assert isinstance(artifact, EditablePrecompileCacheArtifact), (
            "Artifact is not editable"
        )
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> Optional[CacheArtifact]:
        """
        Serialize all backend artifacts with the given key returned in a list.
        """
        result = cls._backend_artifacts_by_key.get(key, None)
        if isinstance(result, EditablePrecompileCacheArtifact):
            result = result.real_encode()
        return result

    @classmethod
    def serialize(cls) -> Optional[tuple[bytes, CacheInfo]]:
        if not cls._dynamo_cache_entries:
            return None

        debug_info = cls.dump_debug_info(
            cls._dynamo_cache_entries, cls._backend_artifacts_by_key
        )
        artifacts = json.dumps({"artifacts": debug_info})
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_cache_save_contents",
                "encoding": "json",
            },
            payload_fn=lambda: artifacts,
            expect_trace_id=False,
        )
        cls._save_artifacts_by_type()

        result = super().serialize()
        assert result is not None
        data, info = result

        return data, info

    @staticmethod
    def dump_debug_info(
        dynamo_entries: dict[str, _DynamoCacheEntry],
        backend_artifacts: dict[
            str, Union[EditablePrecompileCacheArtifact[object], CacheArtifact]
        ],
    ) -> dict[str, Any]:
        """
        Return a JSON serializable debug dump of all entries in the precompile context
        Called in serialize before serialization, and in populate_caches after deserialization
        """
        # Print debug information
        debug_info: defaultdict[str, list[Any]] = defaultdict(list)
        for key, cache_entry in dynamo_entries.items():
            info = cache_entry.debug_info()
            info["key"] = key
            debug_info["precompile_dynamo"].append(info)

        for artifact in backend_artifacts.values():
            if isinstance(artifact, EditablePrecompileCacheArtifact):
                debug_info[artifact.artifact_type].append(artifact.key)
            else:
                debug_info[artifact.__class__.type()].append(artifact.key)

        return debug_info

    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo:
        PrecompileContext._ensure_cache_artifacts_registered()

        backend_artifacts: dict[str, Any] = {}
        dynamo_entries: dict[str, _DynamoCacheEntry] = {}
        cache_info = CacheInfo()
        for artifact in chain(*artifacts.values()):
            if artifact.type() == "autotune":
                # Populate autotune cache artifacts
                artifact.populate_cache()
            elif artifact.type() == "precompile_dynamo":
                assert isinstance(artifact, _DynamoCacheArtifact)
                cache_entry: _DynamoCacheEntry = artifact.after_deserialization()
                dynamo_entries[artifact.key] = cache_entry
            else:
                backend_artifacts[artifact.key] = artifact
            cache_info.add(artifact)

        num_artifacts = len(artifacts["precompile_dynamo"])

        debug_info = PrecompileContext.dump_debug_info(
            dynamo_entries, backend_artifacts
        )
        debug_str = json.dumps(
            {
                "num_entries": num_artifacts,
                "artifacts": debug_info,
            },
        )
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_cache_entries",
                "encoding": "json",
            },
            payload_fn=lambda: debug_str,
            expect_trace_id=False,
        )
        from torch._dynamo.package import _BackendId, DynamoCache

        for key, cache_entry in dynamo_entries.items():
            try:
                backends = cache_entry.backend_ids
                backend_content: dict[_BackendId, PrecompileCacheArtifact[Any]] = {}
                for id_ in backends:
                    if id_ not in backend_artifacts:
                        debug_str = json.dumps(
                            {
                                "entry": cache_entry.debug_info,
                                "key": key,
                            }
                        )
                        logger.warning("Backend not found")
                        torch._logging.trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "dynamo_cache_bypass",
                                "encoding": "json",
                            },
                            payload_fn=lambda: debug_str,
                            expect_trace_id=False,
                        )
                        continue
                    artifact = backend_artifacts[id_]
                    assert isinstance(artifact, PrecompileCacheArtifact)
                    backend_content[id_] = artifact
                DynamoCache.write(cache_entry, backend_content, key)
            except Exception as e:
                logger.warning("Failed to deserialize cache entry %s: %s", key, str(e))

                error = e
                data = json.dumps(
                    {
                        "key": key,
                        "error": str(error),
                    }
                )
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "dynamo_cache_exception",
                        "encoding": "json",
                    },
                    payload_fn=lambda: data,
                )
                continue

        return cache_info

    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            BundledAOTAutogradCacheArtifact,
        )
