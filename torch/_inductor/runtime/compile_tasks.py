from __future__ import annotations

import functools
import io
import linecache
import os
import pickle
import socket
import sys
import threading
import time
import traceback
import warnings
import zlib
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, TYPE_CHECKING

from torch._utils_internal import log_triton_builds


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._inductor.runtime.triton_heuristics import CachingAutotuner


def _reload_python_module(
    key: str, path: str, set_sys_modules: bool = True
) -> ModuleType:
    with open(path) as f:
        try:
            code = compile(f.read(), path, "exec", dont_inherit=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {path}\n{type(e).__name__}: {e}"
            ) from None
        mod = ModuleType(f"{__name__}.{key}")
        mod.__file__ = path
        mod.key = key  # type: ignore[attr-defined]
        exec(code, mod.__dict__, mod.__dict__)
        if set_sys_modules:
            sys.modules[mod.__name__] = mod
        return mod


@functools.cache
def _set_triton_ptxas_path() -> None:
    if os.environ.get("TRITON_PTXAS_PATH") is not None:
        return
    ptxas = Path(__file__).absolute().parents[1] / "bin" / "ptxas"
    if not ptxas.exists():
        return
    if ptxas.is_file() and os.access(ptxas, os.X_OK):
        os.environ["TRITON_PTXAS_PATH"] = str(ptxas)
    else:
        warnings.warn(f"{ptxas} exists but is not an executable")


def _set_triton_libdevice_path() -> None:
    """
    Use the CUDA toolkit's libdevice instead of Triton's bundled version.
    This ensures Triton's pow matches CUDA's powf for bitwise precision.
    Gated by config.eager_numerics.use_pytorch_libdevice.
    """
    from torch._inductor import config

    if not config.eager_numerics.use_pytorch_libdevice:
        return

    _set_triton_libdevice_path_impl()


def _set_triton_libdevice_path_impl() -> None:
    try:
        from triton import knobs
    except ImportError:
        return

    env_path = os.environ.get("TRITON_LIBDEVICE_PATH")
    if env_path is not None:
        knobs.nvidia.libdevice_path = env_path
        return

    if knobs.nvidia.libdevice_path is not None:
        return

    try:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            warnings.warn(
                "CUDA_HOME not set; using Triton's bundled libdevice which may "
                "cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
            return
        libdevice = Path(CUDA_HOME) / "nvvm" / "libdevice" / "libdevice.10.bc"
        if libdevice.is_file():
            knobs.nvidia.libdevice_path = str(libdevice)
            # Also set env var so subprocess compile workers inherit it
            os.environ["TRITON_LIBDEVICE_PATH"] = str(libdevice)
        else:
            warnings.warn(
                f"CUDA libdevice not found at {libdevice}; using Triton's bundled "
                "libdevice which may cause minor precision differences in pow operations. "
                "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
                "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
                stacklevel=3,
            )
    except ImportError:
        warnings.warn(
            "torch.utils.cpp_extension not available; using Triton's bundled "
            "libdevice which may cause minor precision differences in pow operations. "
            "To fix: set TRITON_LIBDEVICE_PATH to your CUDA toolkit's libdevice, "
            "e.g., export TRITON_LIBDEVICE_PATH=/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
            stacklevel=3,
        )


def _worker_compile_triton(
    load_kernel: Callable[[], CachingAutotuner],
    extra_env: dict[str, str],
    extra_config: dict[str, Any],
    streaming_address: str | None = None,
    streaming_kernel_id: bytes | None = None,
) -> tuple[CachingAutotuner | None, int]:
    """Worker entry point for ``AsyncCompile.triton``. Two flows:

    - Streaming (``streaming_address`` non-None): connect to the parent's
      shared listener, send ``streaming_kernel_id`` so the dispatcher routes
      this connection to the right Future, send the kernel, then stream each
      ``CompileResult``. Returns ``(None, elapsed_us)``.
    - Blocking (None): ``precompile(warm_cache_only=True)`` runs every
      config, kernel is pickled back with ``compile_results`` populated.
      Returns ``(kernel, elapsed_us)``.
    """
    _set_triton_ptxas_path()
    os.environ.update(extra_env)
    # Set libdevice path if passed via env from main process
    libdevice_path = extra_env.get("TRITON_LIBDEVICE_PATH")
    if libdevice_path:
        try:
            from triton import knobs

            knobs.nvidia.libdevice_path = libdevice_path
        except ImportError:
            pass
    from torch._inductor import config

    with config.patch(extra_config):
        fail = None
        try:
            start_ns = time.time_ns()
            kernel = load_kernel()
            if streaming_address is not None:
                assert streaming_kernel_id is not None
                _stream_compile_triton(
                    kernel,
                    streaming_address,
                    streaming_kernel_id,
                )
                elapsed_ns = time.time_ns() - start_ns
                # Kernel was streamed back already; nothing left to
                # return via the future payload.
                linecache.clearcache()
                return None, elapsed_ns // 1000
            kernel.precompile(warm_cache_only=True)
            elapsed_ns = time.time_ns() - start_ns
            kernel.prepare_for_pickle()
            # We can release this memory in the compile subprocesses:
            linecache.clearcache()
            return kernel, elapsed_ns // 1000
        except Exception as e:
            fail = str(e)
            raise
        finally:
            log_triton_builds(fail=fail)


_DYN_KERNEL_MODULE_PREFIX = "torch._inductor.runtime.compile_tasks."

# RLock's type isn't directly importable; capture once via type(RLock()).
# _streaming_persistent_id checks this on every traversed pickle object.
_RLOCK_TYPE = type(threading.RLock())


# Wire sentinel for the worker->parent streaming connection. Round-trips as
# an ``is``-identical singleton through ``_streaming_persistent_id`` /
# ``_streaming_persistent_load``. If you add a second sentinel, update both
# functions in lockstep.
_STREAMING_SKIP = object()


def _streaming_persistent_id(obj: Any) -> object | None:
    """Substitute ``_STREAMING_SKIP`` for things that don't survive pickle to
    the parent. All resolve to ``None`` on the parent; downstream consumers
    either don't read them or handle ``None`` defensively.

    - ``RLock``: pickle refuses these outright.
    - ``ModuleType``: the dyn kernel module isn't in the parent's
      ``sys.modules``; uniform substitution is the simplest correct policy.
    - Raw functions in a dyn kernel module (e.g. ``@triton.jit`` bodies):
      pickle-by-name fails on the parent. Restricted to ``FunctionType``
      so we don't catch ``JITFunction``, which inherits the same
      ``__module__`` but pickles fine via its own class.
    """
    if isinstance(obj, _RLOCK_TYPE):
        return _STREAMING_SKIP
    if isinstance(obj, ModuleType):
        return _STREAMING_SKIP
    if isinstance(obj, FunctionType):
        mod = obj.__module__
        if isinstance(mod, str) and mod.startswith(_DYN_KERNEL_MODULE_PREFIX):
            return _STREAMING_SKIP
    return None


def _streaming_persistent_load(pid: object) -> object:
    """Resolve ids from ``_streaming_persistent_id`` to ``None``;
    raise on anything else (wire format divergence)."""
    if pid is _STREAMING_SKIP:
        return None
    raise pickle.UnpicklingError(f"unsupported persistent id: {pid!r}")


class _StreamingPickler(pickle.Pickler):
    """Pickler honoring ``_streaming_persistent_id`` substitutions."""

    persistent_id = staticmethod(_streaming_persistent_id)


class _StreamingUnpickler(pickle.Unpickler):
    """Parent-side companion to ``_StreamingPickler``.

    Trust model: this unpickler accepts arbitrary classes from the worker.
    That is safe only because the worker is part of our trust domain
    (same UID, our own subprocess); pickle deserialization from an
    untrusted peer would be a remote-code-execution risk.
    """

    persistent_load = staticmethod(_streaming_persistent_load)


# Kernel-emit cubin/asm payloads pickle to ~28 KiB median, ~640 KiB p95;
# zlib level 1 cuts those by ~3-5x at <100 MB/s on the worker, paying for
# itself many times over against the 5+ second aggregate socket-flow-control
# wait observed without compression.
_STREAMING_ZLIB_LEVEL = 1


def _streaming_send(conn: Any, obj: Any) -> None:
    """Pickle ``obj`` via ``_StreamingPickler``, zlib-compress, send_bytes.
    Bypasses ``conn.send``, which would use ``ForkingPickler``. The parent
    is the only reader and uses :func:`_streaming_decode` symmetrically."""
    buf = io.BytesIO()
    _StreamingPickler(buf, protocol=pickle.HIGHEST_PROTOCOL).dump(obj)
    conn.send_bytes(zlib.compress(buf.getvalue(), _STREAMING_ZLIB_LEVEL))


def _streaming_decode(payload: bytes) -> Any:
    """Inverse of :func:`_streaming_send`: zlib-decompress + unpickle. Kept
    here so the wire format (compression level, framing) lives in one file.
    """
    return _StreamingUnpickler(io.BytesIO(zlib.decompress(payload))).load()


def _try_set_sock_buf(sock: socket.socket, opt: int, size: int) -> None:
    """setsockopt that tolerates sandbox/seccomp denials. The kernel silently
    clamps size to the system-wide max."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, opt, size)
    except OSError:
        pass


# ----------------------------------------------------------------------------
# Streaming wire-protocol message types.
#
# Every message sent over the streaming socket is one of:
#   _Kernel(kernel)       -- first message; the worker's CachingAutotuner
#   _Success(result)      -- a successful CompileResult
#   _Failure(...)         -- a per-config compile failure (mirrors what the
#                            non-streaming worker stores on
#                            ``kernel._last_compile_exception``); not fatal
#   _Done()               -- explicit end-of-stream marker
#
# EOF on the socket without a preceding _Done is treated as an anomaly
# (worker died mid-stream). Adding a new variant requires updating the
# parent's dispatch in ``_bg_drain_kernel._iter_results``.
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Kernel:
    kernel: Any  # CachingAutotuner


@dataclass(frozen=True)
class _Success:
    result: Any  # CompileResult


@dataclass(frozen=True)
class _Failure:
    """Per-config compile failure. Carries the exception type+message+
    traceback as strings so that exceptions whose ``__reduce__`` is fragile
    (or whose constructor signature has drifted) still survive the wire."""

    exc_type: str
    exc_msg: str
    traceback: str

    @classmethod
    def from_exc(cls, exc: BaseException) -> _Failure:
        # ``traceback.format_exc()`` only reflects the current except-block;
        # we may be called outside one. Use the exception's own __traceback__.
        tb_str = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        return cls(
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
            traceback=tb_str,
        )

    def as_runtime_error(self) -> RuntimeError:
        """Reconstruct as a ``RuntimeError`` for storage on the parent's
        ``kernel._last_compile_exception`` slot."""
        return RuntimeError(f"{self.exc_type}: {self.exc_msg}\n{self.traceback}")


class _Done:
    """End-of-stream marker. Pickled as a class instance; the parent uses
    ``isinstance(msg, _Done)`` so each unpickled instance is fine."""


def _stream_compile_triton(
    kernel: CachingAutotuner,
    streaming_address: str,
    streaming_kernel_id: bytes,
) -> None:
    """Connect to the parent's shared streaming listener at
    ``streaming_address``, send the 8-byte ``streaming_kernel_id`` so the
    parent's dispatcher can route this connection to the right Future. Send
    a ``_Kernel`` as the first message (so the parent can dispatch before
    any compile finishes), then stream each per-config result as a
    ``_Success`` or ``_Failure``, then ``_Done``. ``_StreamingPickler``
    substitutes unpicklable bits per-send so we never mutate the live
    JITFunction (other in-flight compiles in this worker still need it).
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(streaming_address)
        # Bump SNDBUF to mirror the parent's RCVBUF so the worker can write
        # several CompileResult messages without blocking when the drain
        # thread is briefly busy.
        _try_set_sock_buf(sock, socket.SO_SNDBUF, 4 * 1024 * 1024)
        # Identify ourselves to the parent's dispatcher. ID is fixed-size so
        # the parent reads exactly that many bytes and routes us.
        view = memoryview(streaming_kernel_id)
        while view:
            sent = sock.send(view)
            if sent == 0:
                raise OSError("short write on streaming kernel id")
            view = view[sent:]
        conn = Connection(sock.detach())
    except Exception:
        try:
            sock.close()
        except OSError:
            pass
        raise
    try:
        # No in-flight compiles yet; mutating kernel is safe.
        old_values = kernel.prepare_for_pickle()
        try:
            _streaming_send(conn, _Kernel(kernel))
        finally:
            kernel.restore_after_unpickle(old_values)

        for _cfg, result, exc in kernel._iter_compile_results_tagged(parallel=True):
            if exc is None:
                _streaming_send(conn, _Success(result))
            else:
                _streaming_send(conn, _Failure.from_exc(exc))
        _streaming_send(conn, _Done())
    finally:
        conn.close()
