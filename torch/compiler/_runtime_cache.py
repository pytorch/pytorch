"""Runtime dependency transport for precompiled workers."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING

from torch.compiler._cache import _serialize_single_cache, CacheArtifactManager
from torch.compiler._no_compile import is_compilation_forbidden, no_compilation
from torch.utils._appending_byte_serializer import AppendingByteSerializer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _precompile_error(message: str) -> Exception:
    from torch._precompile import PrecompileError

    return PrecompileError(message)


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

    def unseal(self) -> None:
        self.policy.close()
        self.policy = contextlib.ExitStack()
        self.sealed = False


_capture: _RuntimeCapture | None = None
_capture_lock = threading.Lock()


def _active_capture() -> _RuntimeCapture | None:
    # A forked child inherits the parent's scope object; it is not the child's.
    owner = _capture
    return owner if owner is not None and owner.pid == os.getpid() else None


@contextlib.contextmanager
def capture_runtime() -> Iterator[None]:
    """Own one producer lifetime, including strict validation and cleanup.

    Finalizing the cache seals this scope against further compiler work until the
    outer application cleanup has returned.
    """
    global _capture
    if is_compilation_forbidden():
        raise _precompile_error(
            "precompile.capture_runtime cannot start while compilation is forbidden"
        )
    owner = _RuntimeCapture()
    with _capture_lock:
        if _active_capture() is not None:
            raise _precompile_error(
                "Another precompile runtime capture is already active"
            )
        _capture = owner
    try:
        yield
    finally:
        try:
            owner.policy.close()
        finally:
            with _capture_lock:
                if _capture is owner:
                    _capture = None


def finalize_runtime_cache(
    artifact: bytes | None, write: Callable[[bytes | None], None]
) -> None:
    """Pass the finalized cache artifact to ``write``, then keep the scope sealed.

    Pending async compiles are waited on first, then the scope is sealed so any
    later compile raises, and it is unsealed again if serialization or ``write``
    raises so the caller can retry. The caller must stop compiling before
    calling this: a compile submitted on another thread after the wait but
    before the seal is neither waited on nor rejected.
    """
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _active_capture()
    if owner is None:
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
        # Drain before sealing: compile result callbacks re-check
        # check_compilation_allowed, so draining under the seal rejects them.
        AsyncCompile.drain_pending()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        with _capture_lock:
            owner.seal()
        try:
            finalized = None
            if artifacts:
                serializer = AppendingByteSerializer(
                    serialize_fn=_serialize_single_cache
                )
                serializer.extend(artifacts.items())
                finalized = serializer.to_bytes()
            write(finalized)
        except BaseException:
            with _capture_lock:
                owner.unseal()
            raise


def prepare_runtime_cache(artifact: bytes | None) -> None:
    if artifact is not None and CacheArtifactManager.deserialize(artifact) is None:
        raise RuntimeError(
            "Cannot prepare runtime dependencies from an unreadable precompile cache"
        )
