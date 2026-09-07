import copy
import dataclasses
import json
import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
from torch._dynamo.package import (
    _BackendId,
    _DynamoCacheEntry,
    DynamoCache,
    PrecompileCacheEntry,
)


"""
Classes and implementations related to precompile
"""

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class BackendCacheArtifact(Generic[T]):
    """
    Represents a single serializable backend artifact from a dynamo backend.
    Each BackendCacheArtifact has a key associated with it along with some
    serializable content.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """

    key: str
    content: Any

    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...

    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the contents of the artifact.
        """
        self.content = edit_fn(self.content)


@dataclass
class _EagerGraphSource:
    """A Dynamo eager graph carried as generated source rather than as a GraphModule."""

    code: str
    import_block: str
    body: dict[str, Any]


@dataclass
class _SubgraphBlob:
    """A nested HOP body, carried as a real Graph."""

    blob: bytes


class _SourceGraphModule(torch.nn.Module):
    """An fx GraphModule rebuilt from its generated source, without its Graph.

    GraphModule.__reduce__ deliberately serializes only the generated source and
    recovers the Graph by symbolically re-tracing it. That round trip is lossy for a
    Dynamo graph: a node whose target takes no Proxy (torch.cond, autocast enter/exit,
    _set_grad_enabled) either raises or, worse, executes at load and leaves no node
    behind. Dynamo only ever calls the deserialized graph, so drop the Graph rather
    than re-derive a wrong one.
    """

    # Installed by __init__ through __dict__, which nn.Module's __setattr__ would
    # otherwise intercept; declared so they do not resolve through its __getattr__.
    _src: _EagerGraphSource
    _generated_forward: Callable[..., Any]

    def __init__(self, src: _EagerGraphSource) -> None:
        from torch.fx.graph_module import _forward_from_src

        super().__init__()
        body = dict(src.body)
        body["_modules"] = {
            name: _blob_to_graph_module(sub.blob)
            if isinstance(sub, _SubgraphBlob)
            else sub
            for name, sub in body.get("_modules", {}).items()
        }
        body["_src"] = src
        body["_generated_forward"] = _forward_from_src(src.import_block + src.code, {})
        self.__dict__.update(body)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self._generated_forward(self, *args, **kwargs)

    def _current_src(self) -> _EagerGraphSource:
        # _src.body still aliases the containers this instance was BUILT from,
        # which for a deepcopy are the ORIGINAL's parameter/buffer dicts -- so
        # pickling _src directly would round-trip the original's (possibly
        # since-mutated) tensors into the copy's artifact. Snapshot this
        # instance's own parameter/buffer/submodule containers instead (other
        # nn.Module state still comes from _src), converting live sub-GraphModules back
        # to blobs (GraphModule.__reduce__ is lossy for a Dynamo graph, which
        # is why they travel as blobs in the first place).
        body = {**self._src.body}
        body["_parameters"] = dict(self._parameters)
        body["_buffers"] = dict(self._buffers)
        body["_modules"] = {
            name: _SubgraphBlob(_graph_module_to_blob(sub))
            if isinstance(sub, torch.fx.GraphModule)
            else sub
            for name, sub in self._modules.items()
        }
        return dataclasses.replace(self._src, body=body)

    def __reduce__(self) -> tuple[Any, ...]:
        # Required: an artifact can be re-serialized after having been loaded once,
        # and the default reduction would try to pickle the exec'd forward by
        # reference to a module that does not exist.
        return (_SourceGraphModule, (self._current_src(),))

    def __deepcopy__(self, memo: dict[int, Any]) -> "_SourceGraphModule":
        # Same reason GraphModule defines one: without it deepcopy falls through to
        # __reduce__, which re-execs the source and re-loads every nested body on
        # every copy -- and PrecompileContext.record_artifact copies under
        # no_dispatch(), where rebuilding a fake tensor asserts.
        new = object.__new__(type(self))
        memo[id(self)] = new
        # _src and the exec'd forward are safely shared -- __reduce__ snapshots
        # the instance's own containers via _current_src(), so the shared _src
        # never leaks the original's state into a pickled copy. Everything else
        # nn.Module keeps on an instance is state, hook dicts included, and a
        # copy sharing it lets an update on one copy silently edit the other.
        shared = {"_src": self._src, "_generated_forward": self._generated_forward}
        state = {k: v for k, v in self.__dict__.items() if k not in shared}
        new.__dict__.update(copy.deepcopy(state, memo))
        new.__dict__.update(shared)
        return new


def _subgraph_pickler_options() -> Any:
    from torch.fx._graph_pickler import Options

    # ops_filter: a Dynamo graph calls torch.* python functions, not just aten.
    # node_metadata_key_filter: Dynamo's own bookkeeping holds weakrefs to storages
    # that stock pickle cannot follow, and nothing reads it after deserialization.
    return Options(
        ops_filter=None,
        node_metadata_key_filter=lambda key: key
        not in (
            "example_value",
            "tensor_dict",
            "source_fn_stack",
            "nn_module_stack",
            "fwd_source_fn_stack",
        ),
    )


def _graph_module_to_blob(gm: torch.fx.GraphModule) -> bytes:
    from torch.fx._graph_pickler import GraphPickler

    return GraphPickler.dumps(gm, _subgraph_pickler_options())


def _blob_to_graph_module(blob: bytes) -> torch.fx.GraphModule:
    from torch._subclasses import FakeTensorMode
    from torch.fx._graph_pickler import GraphPickler
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    gm = GraphPickler.loads(blob, FakeTensorMode(shape_env=ShapeEnv()))
    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(f"expected a GraphModule, got {type(gm).__name__}")
    gm.recompile()
    return gm


def _graph_module_to_source(gm: torch.fx.GraphModule) -> _EagerGraphSource:
    from torch.fx._lazy_graph_module import _LazyGraphModule
    from torch.fx.graph_module import _format_import_block
    from torch.package import sys_importer

    python_code = (
        gm._real_recompile() if isinstance(gm, _LazyGraphModule) else gm.recompile()
    )
    body = gm.__dict__.copy()
    body.pop("_graph", None)
    # A HOP body is kept as a Graph: some eager HOP implementations run it through
    # fx.Interpreter, which reads .graph. The top level is only ever called.
    body["_modules"] = {
        name: _SubgraphBlob(_graph_module_to_blob(sub))
        if isinstance(sub, torch.fx.GraphModule)
        else sub
        for name, sub in body.get("_modules", {}).items()
    }
    return _EagerGraphSource(
        gm.code, _format_import_block(python_code.globals, sys_importer), body
    )


def _rebuild_eager_artifact(key: str, src: _EagerGraphSource) -> "EagerCacheArtifact":
    return EagerCacheArtifact(key=key, content=_SourceGraphModule(src).forward)


class EagerCacheArtifact(BackendCacheArtifact[Any]):
    def after_deserialization(self) -> Any:
        return self.content

    def __reduce__(self) -> tuple[Any, ...]:
        gm = getattr(self.content, "__self__", None)
        if not isinstance(gm, torch.fx.GraphModule):
            # The eager backend returns a GraphModuleSerializableCallable instead
            # of a bound forward under torch._functorch.config.force_autograd_cache
            # (backends/debugging.py); it pickles through GraphModule.__reduce__
            # and so has the same lossiness, but precompile never reaches it.
            return (type(self), (self.key, self.content))
        return (_rebuild_eager_artifact, (self.key, _graph_module_to_source(gm)))

    def __deepcopy__(self, memo: dict[int, Any]) -> "EagerCacheArtifact":
        return EagerCacheArtifact(
            key=self.key, content=copy.deepcopy(self.content, memo)
        )


class BypassDynamoCacheEntry(Exception):
    pass


class PrecompileContext:
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

    """

    # Protected by the compile_lock
    # _backend_artifacts_by_key organizes results by the key of each artifact.
    # Each object here must be serializable
    _backend_artifacts_by_key: dict[_BackendId, BackendCacheArtifact[Any]] = {}

    # On call to `serialize()`, all cache artifacts in _dynamo_cache_entries are converted
    # into DynamoCacheArtifacts and added to _new_cache_artifacts for serialization
    _dynamo_cache_entries: dict[str, _DynamoCacheEntry] = {}

    @classmethod
    def clear(cls) -> None:
        cls._backend_artifacts_by_key.clear()
        cls._dynamo_cache_entries.clear()

    @classmethod
    def record_artifact(
        cls,
        artifact: BackendCacheArtifact[Any],
    ) -> None:
        """
        Records a backend artifact to be used with dynamo cache entries
        """
        # Temporarily disable all dispatch modes (including FakeTensorMode) during
        # deepcopy to avoid issues with cloning fake tensors (e.g., device mesh
        # with meta tensors that fail when cloning due to device mismatches)
        from torch.utils._mode_utils import no_dispatch

        with no_dispatch():
            cls._backend_artifacts_by_key[_BackendId(artifact.key)] = copy.deepcopy(
                artifact
            )

    @classmethod
    def record_dynamo_cache_entry(
        cls, cache_entry: _DynamoCacheEntry, key: str
    ) -> None:
        cls._dynamo_cache_entries[key] = cache_entry

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        if key not in cls._backend_artifacts_by_key:
            raise AssertionError(f"Key {key} not found in artifacts")
        artifact = cls._backend_artifacts_by_key[_BackendId(key)]
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> BackendCacheArtifact[Any] | None:
        """
        Return the backend cache artifact with the associated key
        """
        return cls._backend_artifacts_by_key.get(_BackendId(key), None)

    @classmethod
    def take_artifact(cls, key: str) -> BackendCacheArtifact[Any] | None:
        """Remove and return one artifact from the process-global staging area."""
        return cls._backend_artifacts_by_key.pop(_BackendId(key), None)

    @staticmethod
    def dump_debug_info(
        dynamo_entries: dict[str, _DynamoCacheEntry],
        backend_artifacts: dict[_BackendId, BackendCacheArtifact[Any]],
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
            debug_info["dynamo"].append(info)

        for artifact in backend_artifacts.values():
            debug_info["backends"].append(artifact.key)

        return debug_info

    @classmethod
    def save_to_dynamo_cache(cls) -> dict[str, Any]:
        precompile_cache_entries, debug_info = cls.create_cache_entries()
        for key, entry in precompile_cache_entries.items():
            DynamoCache.write(entry, key)
        return debug_info

    @classmethod
    def create_cache_entries(
        cls,
    ) -> tuple[dict[str, PrecompileCacheEntry], dict[str, Any]]:
        """
        Grabs all the cache entries in the precompile context and
        stitches them together into full PrecompileCacheEntries.
        """
        dynamo_entries = cls._dynamo_cache_entries
        backend_artifacts = cls._backend_artifacts_by_key

        num_artifacts = len(dynamo_entries)

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

        precompile_cache_entries = {}

        for key, cache_entry in dynamo_entries.items():
            try:
                result = PrecompileCacheEntry.from_cache_entry(
                    cache_entry, backend_artifacts
                )
                if result is not None:
                    precompile_cache_entries[key] = result
            except Exception as e:
                logger.warning("Failed to create cache entry %s", key, exc_info=True)

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
        return precompile_cache_entries, debug_info
