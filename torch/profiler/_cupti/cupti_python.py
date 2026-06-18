# mypy: allow-untyped-defs
"""CUPTI type and constant definitions used by the in-process monitor.

This module reaches into CUPTI's ABI constants. It does not re-export
cupti-python's types; callers import those directly
(``from cupti.cupti import ActivityKind``). The enums this module itself needs at
runtime are imported eagerly below. cupti-python is a hard requirement, so
importing this module without it raises ``ModuleNotFoundError`` (catchable by
optional consumers).
"""

from __future__ import annotations

import os


# cupti-python enums used at runtime. cupti-python is a hard requirement of this
# module; its absence surfaces as a catchable ModuleNotFoundError for optional
# consumers (e.g. those probing the monitor import).
try:
    from cupti.cupti import (  # pyrefly: ignore[missing-import]
        Driver_api_trace_cbid,
        Runtime_api_trace_cbid,
    )
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "torch.profiler._cupti requires the cupti-python package. "
        "Install cupti-python to use the experimental CUPTI monitor."
    ) from exc


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


def disabled_runtime_cbids() -> tuple[int, ...]:
    """Runtime API callbacks filtered out of activity to cut trace volume."""
    cbids = Runtime_api_trace_cbid
    return (
        cbids.cudaGetDevice_v3020,
        cbids.cudaSetDevice_v3020,
        cbids.cudaGetLastError_v3020,
        cbids.cudaEventCreate_v3020,
        cbids.cudaEventCreateWithFlags_v3020,
        cbids.cudaEventDestroy_v3020,
    )


def disabled_driver_cbids() -> tuple[int, ...]:
    """Driver API callbacks filtered out of activity to cut trace volume."""
    cbids = Driver_api_trace_cbid
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
