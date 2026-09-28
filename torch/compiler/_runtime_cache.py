"""Runtime dependency transport for precompiled workers."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING

from torch.compiler._cache import (
    _serialize_single_cache,
    CacheArtifactManager,
)
from torch.compiler._no_compile import is_compilation_forbidden, no_compilation
from torch.utils._appending_byte_serializer import AppendingByteSerializer


if TYPE_CHECKING:
    from collections.abc import Iterator


class _RuntimeCapture:
    def __init__(self) -> None:
        self.pid = os.getpid()
        self.sealed = False
        self.policy = contextlib.ExitStack()
        self.finalize_lock = threading.Lock()

    def seal(self) -> None:
        if self.sealed:
            raise RuntimeError(
                "The precompile runtime cache has already been finalized"
            )
        self.policy.enter_context(no_compilation())
        self.sealed = True


_capture: _RuntimeCapture | None = None
_capture_lock = threading.Lock()


@contextlib.contextmanager
def capture_runtime() -> Iterator[None]:
    """Own one producer lifetime, including strict validation and cleanup.

    Finalizing the cache seals this scope against further compiler work until the
    outer application cleanup has returned.
    """
    global _capture
    if is_compilation_forbidden():
        raise RuntimeError(
            "Runtime capture cannot start while compilation is forbidden"
        )
    owner = _RuntimeCapture()
    with _capture_lock:
        if _capture is not None:
            raise RuntimeError("Another precompile runtime capture is already active")
        _capture = owner
    try:
        yield
    finally:
        try:
            owner.policy.close()
        finally:
            with _capture_lock:
                _capture = None


def finalize_runtime_cache(artifact: bytes | None) -> bytes:
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _capture
    if owner is None or owner.pid != os.getpid():
        raise RuntimeError(
            "precompile.finalize_cache requires the worker's capture_runtime scope"
        )
    with owner.finalize_lock:
        if owner.sealed:
            raise RuntimeError(
                "The precompile runtime cache has already been finalized"
            )
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot finalize an unreadable precompile cache")
        AsyncCompile.drain_pending()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        with _capture_lock:
            owner.seal()
        serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
        serializer.extend(artifacts.items())
        return serializer.to_bytes()


def prepare_runtime_cache(artifact: bytes | None) -> None:
    if artifact is not None and CacheArtifactManager.deserialize(artifact) is None:
        raise RuntimeError(
            "Cannot prepare runtime dependencies from an unreadable precompile cache"
        )
