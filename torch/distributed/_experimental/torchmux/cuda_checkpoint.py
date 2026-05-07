"""Thin ctypes wrapper around the CUDA driver's process-checkpoint APIs.

These are the ``cuCheckpointProcess*`` entry points exported by
``libcuda.so.1`` (driver >= 555). A process can checkpoint itself (VRAM
gets paged to driver-managed host buffers and the GPU context is torn
down) and later restore (VRAM comes back with pointers intact).

The CLI ``cuda-checkpoint`` is NOT required — we call the driver APIs
directly.

Exposed API:
  - :func:`checkpoint_self` — lock + checkpoint the current process.
  - :func:`restore_self`    — restore + unlock.
  - :func:`get_state`       — query the current process state.
  - :class:`CudaCheckpointError` — raised on non-zero CUresult.
"""

import ctypes
import os


_CUDA = None


def _cuda() -> ctypes.CDLL:
    global _CUDA
    if _CUDA is None:
        _CUDA = ctypes.CDLL("libcuda.so.1")
        for name in (
            "cuCheckpointProcessLock",
            "cuCheckpointProcessCheckpoint",
            "cuCheckpointProcessRestore",
            "cuCheckpointProcessUnlock",
            "cuCheckpointProcessGetState",
        ):
            fn = getattr(_CUDA, name)
            fn.restype = ctypes.c_int
            if name == "cuCheckpointProcessGetState":
                fn.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_int)]
            else:
                fn.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    return _CUDA


class CudaCheckpointError(RuntimeError):
    pass


def _call(fn_name: str) -> None:
    pid = os.getpid()
    fn = getattr(_cuda(), fn_name)
    rc = fn(pid, None)
    if rc != 0:
        raise CudaCheckpointError(f"{fn_name}(pid={pid}) failed with CUresult={rc}")


def get_state() -> int:
    """Return the CUDA process state int (0=running, 1=locked, 2=checkpointed, 3=failed)."""
    state = ctypes.c_int(-1)
    rc = _cuda().cuCheckpointProcessGetState(os.getpid(), ctypes.byref(state))
    if rc != 0:
        raise CudaCheckpointError(
            f"cuCheckpointProcessGetState failed with CUresult={rc}"
        )
    return state.value


def checkpoint_self() -> None:
    """Lock + checkpoint the current process. Releases VRAM. CUDA ops will
    fail until :func:`restore_self` is called.

    The checkpointed state (copies of every active device allocation +
    stream / event / module metadata) is written to host buffers the CUDA
    driver owns inside this process's address space. Those buffers are not
    exposed through any public API — the args structs for the
    Checkpoint/Restore calls are ``reserved[8]`` with no output pointer,
    no file path, no callback. If you need to touch the bytes, do not use
    this API; copy tensors to host memory yourself (see
    ``torch.cuda._mem_tracker.serialize``) or go through CRIU.
    """
    _call("cuCheckpointProcessLock")
    try:
        _call("cuCheckpointProcessCheckpoint")
    except BaseException:
        _call("cuCheckpointProcessUnlock")
        raise


def restore_self() -> None:
    """Restore + unlock the current process. Re-allocates VRAM, resumes CUDA ops."""
    # If restore fails then we're in a bad state - don't try to recover, just
    # let the exception propagate and hope nobody catches it.
    _call("cuCheckpointProcessRestore")
    _call("cuCheckpointProcessUnlock")
