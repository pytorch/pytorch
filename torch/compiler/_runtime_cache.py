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
    def __init__(self, *, explicit: bool) -> None:
        self.pid = os.getpid()
        # Opened by capture_runtime(), else by the first open precompile.capture
        # and closed by the last one to exit.
        self.explicit = explicit
        self.captures = 0
        self.sealed = False
        self.policy = contextlib.ExitStack()
        self.freeze_lock = threading.Lock()

    def seal(self) -> None:
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


def _close(owner: _RuntimeCapture) -> None:
    global _capture
    try:
        owner.policy.close()
    finally:
        with _capture_lock:
            if _capture is owner:
                _capture = None


@contextlib.contextmanager
def capture_runtime() -> Iterator[None]:
    """Own one producer lifetime, including strict validation and cleanup.

    The capture inside it seals the scope against further compiler work until
    the outer application cleanup has returned.
    """
    global _capture
    if is_compilation_forbidden():
        raise _precompile_error(
            "precompile.capture_runtime cannot start while compilation is forbidden"
        )
    owner = _RuntimeCapture(explicit=True)
    with _capture_lock:
        if _active_capture() is not None:
            raise _precompile_error(
                "Another precompile runtime capture or an open precompile.capture "
                "is already active"
            )
        _capture = owner
    try:
        yield
    finally:
        _close(owner)


@contextlib.contextmanager
def capture_scope() -> Iterator[None]:
    """The scope a precompile.capture block records into: the enclosing
    capture_runtime() scope, else one shared by the captures open in this process.
    """
    global _capture
    with _capture_lock:
        owner = _active_capture()
        if owner is None:
            owner = _capture = _RuntimeCapture(explicit=False)
        owner.captures += 1
    try:
        yield
    finally:
        with _capture_lock:
            owner.captures -= 1
            last = not owner.explicit and owner.captures == 0
        if last:
            _close(owner)


def seal_runtime_scope() -> None:
    """Seal an enclosing capture_runtime() scope whose cache is already written."""
    owner = _active_capture()
    if owner is None or not owner.explicit:
        return
    with owner.freeze_lock, _capture_lock:
        if not owner.sealed:
            owner.seal()


def freeze_runtime_cache(
    artifact: bytes | None,
    write: Callable[[bytes | None], None],
    *,
    final: bool,
    compiled: bool,
) -> None:
    """Pass the cache artifact, with the scope's runtime dependencies, to ``write``.

    Pending async compiles are waited on first. A ``final`` write inside a
    capture_runtime() scope then seals it so any later compile raises, and keeps
    it sealed unless serialization or ``write`` raises. A capture that did not
    run eagerly (``compiled``) but saved no artifact has no dependencies to
    carry; a capture_runtime() scope refuses it, since strict precompile.load
    could not serve it either. The caller must stop compiling before calling
    this: a compile submitted on another thread after the wait but before the
    seal is neither waited on nor rejected.
    """
    import torch
    from torch._inductor.async_compile import AsyncCompile

    owner = _active_capture()
    if owner is None:
        raise RuntimeError("precompile.capture has no runtime capture scope")
    if compiled and artifact is None:
        if owner.explicit:
            raise RuntimeError(
                "precompile.capture_runtime requires the compiled cache artifact of "
                "a compiled capture, and this capture saved none"
            )
        write(None)
        return
    with owner.freeze_lock:
        if owner.sealed:
            raise RuntimeError(
                "The precompile.capture_runtime scope was already sealed by an "
                "earlier capture"
            )
        artifacts = (
            CacheArtifactManager.deserialize(artifact) if artifact is not None else {}
        )
        if artifacts is None:
            raise RuntimeError("Cannot freeze an unreadable precompile cache")
        # Drain before sealing: compile result callbacks re-check
        # check_compilation_allowed, so draining under the seal rejects them.
        AsyncCompile.drain_pending()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        seal = final and owner.explicit
        if seal:
            with _capture_lock:
                owner.seal()
        try:
            frozen = None
            if artifacts:
                serializer = AppendingByteSerializer(
                    serialize_fn=_serialize_single_cache
                )
                serializer.extend(artifacts.items())
                frozen = serializer.to_bytes()
            write(frozen)
        except BaseException:
            if seal:
                with _capture_lock:
                    owner.unseal()
            raise


def prepare_runtime_cache(artifact: bytes | None) -> None:
    if artifact is not None and CacheArtifactManager.deserialize(artifact) is None:
        raise RuntimeError(
            "Cannot prepare runtime dependencies from an unreadable precompile cache"
        )
