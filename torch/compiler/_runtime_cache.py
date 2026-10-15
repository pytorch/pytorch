"""Runtime dependency transport for precompiled workers."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import TYPE_CHECKING

from torch.compiler._cache import _serialize_single_cache, CacheArtifactManager
from torch.utils._appending_byte_serializer import AppendingByteSerializer


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class _RuntimeCapture:
    def __init__(self) -> None:
        self.pid = os.getpid()
        # The precompile.capture blocks open in this process; the last to exit
        # closes the scope.
        self.captures = 0
        self.freeze_lock = threading.Lock()


_capture: _RuntimeCapture | None = None
_capture_lock = threading.Lock()


def _active_capture() -> _RuntimeCapture | None:
    # A forked child inherits the parent's scope object; it is not the child's.
    owner = _capture
    return owner if owner is not None and owner.pid == os.getpid() else None


@contextlib.contextmanager
def capture_scope() -> Iterator[None]:
    """The scope a precompile.capture block records into, shared by the
    captures open in this process and closed by the last one to exit.
    """
    global _capture
    with _capture_lock:
        owner = _active_capture()
        if owner is None:
            owner = _capture = _RuntimeCapture()
        owner.captures += 1
    try:
        yield
    finally:
        with _capture_lock:
            owner.captures -= 1
            if owner.captures == 0:
                _capture = None


def freeze_runtime_cache(
    artifact: bytes | None, write: Callable[[bytes | None], None], *, compiled: bool
) -> None:
    """Pass the cache artifact, with the scope's runtime dependencies, to ``write``.

    Pending async compiles are waited on first. A capture that did not run
    eagerly (``compiled``) but saved no artifact raises, since precompile.load
    could not serve it without compiling. The caller must stop compiling before
    calling this: a compile submitted on another thread after the wait is not
    waited on.
    """
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _active_capture()
    if owner is None:
        raise RuntimeError("precompile.capture has no runtime capture scope")
    if compiled and artifact is None:
        raise RuntimeError(
            "precompile.capture requires the cache artifact of a compiled "
            "capture, and this capture saved none"
        )
    with owner.freeze_lock:
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot freeze an unreadable precompile cache")
        # Another open capture may still be compiling, so keep the workers.
        AsyncCompile.drain_pending(shutdown=False)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        frozen = None
        if artifacts:
            serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
            serializer.extend(artifacts.items())
            frozen = serializer.to_bytes()
        write(frozen)
