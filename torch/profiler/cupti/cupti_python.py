# mypy: allow-untyped-defs
"""CUPTI type and constant definitions used by the in-process monitor.

This module is the single place that reaches into the ``cupti-python`` package
and into CUPTI's ABI constants. The cupti-python module is imported lazily and
cached by :func:`_cupti`, and the activity kinds, enum values and record classes
the monitor needs are re-exported as module attributes resolved on first access
(see :func:`__getattr__`). Callers therefore just import what they use --
``from .cupti_python import ActivityKind, ActivityKernel11`` -- and write
``ActivityKind.MEMCPY`` directly (cupti-python's enums are ``IntEnum``, so kind
members compare equal to the raw integer kind read from a record), while the
cupti import stays deferred until one of those names is first touched.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


# Environment override for the libcupti to dlopen; see find_cupti_library().
LIBCUPTI_PATH_ENV = "TORCH_CUPTI_MONITOR_LIBCUPTI_PATH"

# CUPTI C-API result/flag constants (cupti_result.h / cupti_activity.h). These
# are stable ABI values, so they are spelled out rather than resolved.
CUPTI_SUCCESS = 0
CUPTI_ERROR_MAX_LIMIT_REACHED = 12
CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1

# CUPTI overhead-kind codes (CUpti_ActivityOverheadKind in cupti_activity.h).
# cupti-python does not surface these as an enum, so the code -> name mapping is
# mirrored here.
OVERHEAD_KIND_NAMES: dict[int, str] = {
    0: "Unknown",
    1: "Driver Compiler",
    1 << 16: "Buffer Flush",
    2 << 16: "Instrumentation",
    3 << 16: "Resource",
    4 << 16: "Runtime Triggered Module Loading",
    5 << 16: "Lazy Function Loading",
    6 << 16: "Command Buffer Full",
    7 << 16: "Activity Buffer Request",
    8 << 16: "UVM Activity Init",
}

# cupti-python names re-exported lazily by __getattr__: the enums and the record
# classes the monitor decodes. Resolved off the cupti module on first access.
_REEXPORTED = frozenset(
    {
        "ActivityKind",
        "ExternalCorrelationKind",
        "Runtime_api_trace_cbid",
        "Driver_api_trace_cbid",
        "ActivityKernel11",
        "ActivityMemcpy6",
        "ActivityMemset4",
        "ActivityAPI",
        "ActivityExternalCorrelation",
        "ActivityOverhead3",
        "ActivityCudaEvent2",
        "ActivitySynchronization2",
    }
)


@lru_cache(maxsize=1)
def _cupti() -> Any:
    """Import and return the cupti-python module (once, cached)."""
    try:
        from cupti import cupti as cc  # pyrefly: ignore[missing-import]
    except ModuleNotFoundError as exc:
        # Keep this a ModuleNotFoundError (not a bare ImportError) so optional
        # consumers that probe `import torch.profiler.cupti.monitor` degrade
        # gracefully when cupti-python is absent.
        raise ModuleNotFoundError(
            "torch.profiler.cupti requires the cupti-python package. "
            "Install cupti-python to use the experimental CUPTI monitor."
        ) from exc

    return cc


def __getattr__(name: str) -> Any:
    # Lazily expose cupti-python's enums and record classes as attributes of
    # this module, so callers can `from .cupti_python import ActivityKind` and
    # use `ActivityKind.MEMCPY` directly while the cupti import stays deferred.
    if name in _REEXPORTED:
        return getattr(_cupti(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@lru_cache(maxsize=1)
def disabled_runtime_cbids() -> tuple[int, ...]:
    """Runtime API callbacks filtered out of activity to cut trace volume."""
    cbids = _cupti().Runtime_api_trace_cbid
    return (
        cbids.cudaGetDevice_v3020,
        cbids.cudaSetDevice_v3020,
        cbids.cudaGetLastError_v3020,
        cbids.cudaEventCreate_v3020,
        cbids.cudaEventCreateWithFlags_v3020,
        cbids.cudaEventDestroy_v3020,
    )


@lru_cache(maxsize=1)
def disabled_driver_cbids() -> tuple[int, ...]:
    """Driver API callbacks filtered out of activity to cut trace volume."""
    cbids = _cupti().Driver_api_trace_cbid
    return (
        cbids.cuKernelGetAttribute,
        cbids.cuDevicePrimaryCtxGetState,
        cbids.cuCtxGetCurrent,
    )


def find_cupti_library() -> str:
    """Resolve the libcupti shared object to dlopen for the CUPTI v2 API.

    Honors the LIBCUPTI_PATH_ENV override, otherwise resolves via cuda
    pathfinder -- the same mechanism cupti-python and torch use, so we share the
    single libcupti already loaded in the process. Diverging here (e.g.
    preferring a newer site-packages wheel for the v2 API) would create a second
    CUPTI instance that collides with the stock profiler's subscriber
    (CUPTI_ERROR_MULTIPLE_SUBSCRIBERS). Reaching a different libcupti has to be
    done at load time (e.g. LD_PRELOAD) so every consumer agrees on one.
    """
    override = os.environ.get(LIBCUPTI_PATH_ENV)
    if override:
        return override
    from cuda.pathfinder import (  # pyrefly: ignore[missing-import]
        load_nvidia_dynamic_lib,
    )

    path = load_nvidia_dynamic_lib("cupti").abs_path
    if path is None:
        raise RuntimeError("cuda pathfinder could not resolve a libcupti path")
    return path
