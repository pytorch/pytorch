import torch
import dataclasses
from itertools import chain
from typing_extensions import override
from typing import Callable, Optional
from torch.compiler._cache import CacheArtifact, CacheArtifactManager, CacheArtifactsResult,  CacheInfo
"""
Classes and implementations related to precompile
"""


class PrecompileCacheArtifact(CacheArtifact):
    """
    Data for each cache artifact that will be serialized and deserialized by
    PrecompileContext, rather than CacheArtifactManager.
    """
    @override
    def populate_cache(self) -> None:
        raise RuntimeError("Precompile cache artifacts do not populate caches")

    @override
    def precompile_compatible(self) -> bool:
        return True


class PrecompileContext(CacheArtifactManager):
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently:
    specifically, it returns a full Callable object instead of populating caches.

    Only the following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact

    TODO: interface not yet finalized
    """


    @staticmethod
    def deserialize_into_callable(serialized_artifacts: bytes) -> Optional[Callable]:
        """
        Similar to CacheArtifactManager.serialize, but instead of populating caches,
        we want to return a full Callable object
        """
        artifacts : Optional[CacheArtifactsResult] = PrecompileContext.deserialize(serialized_artifacts)
        if not artifacts:
            return None
        for artifact in chain(*artifacts.values()):
            if not artifact.precompile_compatible():
                raise RuntimeError(f"Unsupported precompile artifact type: {artifact.__class__.type()}")
        pass # TODO

    @staticmethod
    def populate_caches(artifacts) -> CacheInfo:
        raise RuntimeError("Precompile Contexts do not directly populate any caches")


    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        from torch._functorch._aot_autograd.autograd_cache import (  # noqa: F401
            BundledAOTAutogradCacheArtifact,
        )
