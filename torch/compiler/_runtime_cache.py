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
        # The precompile.capture blocks open in this process. Overlapping
        # captures are unsupported; counting them only keeps a misuse from
        # closing the scope under the other capture.
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
    """The scope a precompile.capture block records into, closed when the
    block exits. Overlapping captures are unsupported; a misused overlap shares
    the scope rather than closing it under the other capture.
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
    snapshot: Callable[[], bytes | None], write: Callable[[bytes | None], None]
) -> None:
    """Pass the cache artifact, with the scope's runtime dependencies, to ``write``.

    Pending async compiles are waited on before ``snapshot`` renders the
    artifact, so what they record is in it. The caller must stop compiling
    before calling this: a compile submitted on another thread after the wait is
    not waited on.
    """
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _active_capture()
    if owner is None:
        raise RuntimeError("precompile.capture has no runtime capture scope")
    with owner.freeze_lock:
        # Keep the workers: a save() checkpoint is followed by more of the block.
        AsyncCompile.drain_pending(shutdown=False)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        artifact = snapshot()
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot freeze an unreadable precompile cache")
        frozen = None
        if artifacts:
            serializer = AppendingByteSerializer(serialize_fn=_serialize_single_cache)
            serializer.extend(artifacts.items())
            frozen = serializer.to_bytes()
        write(frozen)
