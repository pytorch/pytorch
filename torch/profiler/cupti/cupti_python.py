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

import ctypes
import os
from collections.abc import Iterable  # noqa: TC003
from functools import lru_cache
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    # cupti-python's enum classes, for typing the pylibcupti method signatures.
    # (Resolved to Any by pyrefly -- the cupti stub is a missing-import -- but the
    # signatures still read as the intended CUPTI types.)
    from cupti.cupti import (  # pyrefly: ignore[missing-import]
        ActivityKind,
        Driver_api_trace_cbid,
        ExternalCorrelationKind,
        Runtime_api_trace_cbid,
    )


# Environment override for the libcupti to dlopen; see find_cupti_library().
LIBCUPTI_PATH_ENV = "TORCH_CUPTI_MONITOR_LIBCUPTI_PATH"

# CUPTI C-API result/flag constants (cupti_result.h / cupti_activity.h). These
# are stable ABI values, so they are spelled out rather than resolved.
CUPTI_SUCCESS = 0
CUPTI_ERROR_MAX_LIMIT_REACHED = 12
CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1

# CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_USER_DEFINED_RECORDS (not surfaced
# by cupti-python); set on the subscription to turn on the v2 user-defined-record
# path. Minimum libcupti version with that API is CUPTI 13.2.
_ATTR_USER_DEFINED_RECORDS = 11
_MIN_V2_VERSION = 130200

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


# Record class -> the cupti-python numpy dtype mirroring its C struct. The dtype's
# itemsize is the record's byte size, i.e. its stride in a v1 activity buffer
# (records are whole structs packed back-to-back), which lets the v1 demux address
# records directly instead of walking the cuptiActivityGetNextRecord cursor.
_RECORD_DTYPE_NAMES = {
    "ActivityKernel11": "activity_kernel11_dtype",
    "ActivityMemcpy6": "activity_memcpy6_dtype",
    "ActivityMemset4": "activity_memset4_dtype",
    "ActivityAPI": "activity_api_dtype",
    "ActivityExternalCorrelation": "activity_external_correlation_dtype",
    "ActivityOverhead3": "activity_overhead3_dtype",
    "ActivityCudaEvent2": "activity_cuda_event2_dtype",
    "ActivitySynchronization2": "activity_synchronization2_dtype",
}


@lru_cache(maxsize=1)
def record_struct_sizes() -> dict[str, int]:
    """Record-class name -> its C-struct byte size (the cupti-python dtype
    itemsize) -- the record's stride in a v1 activity buffer."""
    import numpy as np

    cc = _cupti()
    return {
        cls: int(np.dtype(getattr(cc, dtype)).itemsize)
        for cls, dtype in _RECORD_DTYPE_NAMES.items()
    }


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


def _configure_ctypes(lib: ctypes.CDLL) -> None:
    lib.cuptiGetVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    lib.cuptiGetVersion.restype = ctypes.c_int
    lib.cuptiGetTimestamp.argtypes = [ctypes.POINTER(ctypes.c_uint64)]
    lib.cuptiGetTimestamp.restype = ctypes.c_int
    lib.cuptiActivityRegisterCallbacks.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
    lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
    lib.cuptiActivityEnable.restype = ctypes.c_int
    lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
    lib.cuptiActivityDisable.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityEnableRuntimeApi"):
        lib.cuptiActivityEnableRuntimeApi.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint8,
        ]
        lib.cuptiActivityEnableRuntimeApi.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityEnableDriverApi"):
        lib.cuptiActivityEnableDriverApi.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint8,
        ]
        lib.cuptiActivityEnableDriverApi.restype = ctypes.c_int
    lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
    lib.cuptiActivityFlushAll.restype = ctypes.c_int
    lib.cuptiActivityGetNextRecord.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    lib.cuptiActivityGetNextRecord.restype = ctypes.c_int
    lib.cuptiActivityGetNumDroppedRecords.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cuptiActivityGetNumDroppedRecords.restype = ctypes.c_int
    lib.cuptiActivityEnableHWTrace.argtypes = [ctypes.c_uint8]
    lib.cuptiActivityEnableHWTrace.restype = ctypes.c_int
    lib.cuptiActivityRegisterTimestampCallback.argtypes = [ctypes.c_void_p]
    lib.cuptiActivityRegisterTimestampCallback.restype = ctypes.c_int
    lib.cuptiActivityPushExternalCorrelationId.argtypes = [
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    lib.cuptiActivityPushExternalCorrelationId.restype = ctypes.c_int
    lib.cuptiActivityPopExternalCorrelationId.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.cuptiActivityPopExternalCorrelationId.restype = ctypes.c_int
    lib.cuptiGetResultString.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.cuptiGetResultString.restype = ctypes.c_int

    # v2 / user-defined-record API -- only present in libcupti >= 13.2; guarded
    # so configuring against an older libcupti (v1 only) still succeeds.
    if hasattr(lib, "cuptiSubscribe_v2"):
        lib.cuptiSubscribe_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle* subscriber
            _CB_FUNC,  # CUpti_CallbackFunc callback
            ctypes.c_void_p,  # void* userdata
            ctypes.POINTER(_SubscriberParams),  # CUpti_SubscriberParams* pParams
        ]
        lib.cuptiSubscribe_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiUnsubscribe"):
        lib.cuptiUnsubscribe.argtypes = [ctypes.c_void_p]
        lib.cuptiUnsubscribe.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivitySetAttribute_v2"):
        lib.cuptiActivitySetAttribute_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        lib.cuptiActivitySetAttribute_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityRegisterCallbacks_v2"):
        lib.cuptiActivityRegisterCallbacks_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        lib.cuptiActivityRegisterCallbacks_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityEnable_v2"):
        lib.cuptiActivityEnable_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        lib.cuptiActivityEnable_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityDisable_v2"):
        lib.cuptiActivityDisable_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.cuptiActivityDisable_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityPushExternalCorrelationId_v2"):
        lib.cuptiActivityPushExternalCorrelationId_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        lib.cuptiActivityPushExternalCorrelationId_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityPopExternalCorrelationId_v2"):
        lib.cuptiActivityPopExternalCorrelationId_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.cuptiActivityPopExternalCorrelationId_v2.restype = ctypes.c_int


# --- v2 user-defined-record ctypes structs --------------------------------
class _SubscriberParams(ctypes.Structure):
    """Mirror of CUpti_SubscriberParams (cupti_callbacks.h, CUPTI >= 13.2), the
    4th argument to cuptiSubscribe_v2. Member order/types must match CUPTI."""

    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("subscriberName", ctypes.c_char_p),
        ("oldSubscriberName", ctypes.c_char_p),
        ("oldSubscriberSize", ctypes.c_size_t),
        ("allowMultipleSubscribers", ctypes.c_uint8),
        ("padding", ctypes.c_uint8 * 7),
    ]


class _UDFieldSelection(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("numFields", ctypes.c_size_t),
        ("pFieldIds", ctypes.POINTER(ctypes.c_int)),
    ]


class _UDActivityConfig(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("fieldSelection", _UDFieldSelection),
    ]


# cuptiSubscribe requires a valid CUpti_CallbackFunc, but the monitor drives
# collection through the activity API, not callbacks -- a no-op suffices. Kept
# alive process-wide so the ctypes trampoline isn't garbage-collected.
_CB_FUNC = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p
)


def _noop_callback(*_args: object) -> None:
    pass


_NOOP_CB = _CB_FUNC(_noop_callback)


class CuptiError(RuntimeError):
    pass


class _PyLibCupti:
    """Pythonic wrapper over libcupti's CUPTI Activity API: each method hides the
    ctypes marshalling and rc-checking behind a clean call, so callers (the
    monitor, enable_hes_early, the v2 path) never touch ctypes. Get the
    process-wide instance via :func:`pylibcupti`. Methods that must succeed raise
    CuptiError; genuinely-optional ones return a bool / None."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self._lib = lib

    def _result_string(self, rc: int) -> str:
        result = ctypes.c_char_p()
        rc2 = self._lib.cuptiGetResultString(rc, ctypes.byref(result))
        if rc2 == CUPTI_SUCCESS and result.value is not None:
            return result.value.decode()
        return f"rc={rc}"

    def _check(self, rc: int, name: str) -> None:
        if rc != CUPTI_SUCCESS:
            raise CuptiError(f"{name} failed with {self._result_string(rc)}")

    def get_version(self) -> int:
        version = ctypes.c_uint32()
        self._check(self._lib.cuptiGetVersion(ctypes.byref(version)), "cuptiGetVersion")
        return version.value

    def get_timestamp(self) -> int:
        """CUPTI's normalized nanosecond clock -- the same timebase as activity
        record START/END timestamps, so a value captured here is directly
        comparable to decoded record timestamps."""
        ts = ctypes.c_uint64()
        self._check(self._lib.cuptiGetTimestamp(ctypes.byref(ts)), "cuptiGetTimestamp")
        return ts.value

    def activity_enable(self, kind: ActivityKind) -> None:
        self._check(self._lib.cuptiActivityEnable(kind), "cuptiActivityEnable")

    def activity_disable(self, kind: ActivityKind) -> None:
        self._check(self._lib.cuptiActivityDisable(kind), "cuptiActivityDisable")

    def activity_enable_runtime_api(
        self, cbid: Runtime_api_trace_cbid, enabled: bool
    ) -> bool:
        """Enable/disable a single runtime-API cbid. Returns False (no-op) if the
        libcupti lacks the per-cbid filter symbol."""
        if not hasattr(self._lib, "cuptiActivityEnableRuntimeApi"):
            return False
        self._check(
            self._lib.cuptiActivityEnableRuntimeApi(cbid, 1 if enabled else 0),
            "cuptiActivityEnableRuntimeApi",
        )
        return True

    def activity_enable_driver_api(
        self, cbid: Driver_api_trace_cbid, enabled: bool
    ) -> bool:
        """Enable/disable a single driver-API cbid. Returns False (no-op) if the
        libcupti lacks the per-cbid filter symbol."""
        if not hasattr(self._lib, "cuptiActivityEnableDriverApi"):
            return False
        self._check(
            self._lib.cuptiActivityEnableDriverApi(cbid, 1 if enabled else 0),
            "cuptiActivityEnableDriverApi",
        )
        return True

    def activity_flush_all(self, forced: bool) -> None:
        flag = CUPTI_ACTIVITY_FLAG_FLUSH_FORCED if forced else 0
        self._check(self._lib.cuptiActivityFlushAll(flag), "cuptiActivityFlushAll")

    def activity_get_next_record(
        self, buffer_ptr: int, valid_size: int, prev_record: int | None = None
    ) -> int | None:
        """The address of the activity record after ``prev_record`` in the buffer
        (or the first record when ``prev_record`` is None), or None once the buffer
        is exhausted. The record pointer is in/out: it must carry the previous
        record back in to advance -- passing NULL each call would re-return the
        first record forever (an infinite walk)."""
        record_ptr = ctypes.c_void_p(prev_record)
        rc = self._lib.cuptiActivityGetNextRecord(
            ctypes.c_void_p(buffer_ptr),
            ctypes.c_size_t(valid_size),
            ctypes.byref(record_ptr),
        )
        if rc == CUPTI_SUCCESS:
            if record_ptr.value is None:
                raise CuptiError("CUPTI returned null activity record pointer")
            return record_ptr.value
        if rc == CUPTI_ERROR_MAX_LIMIT_REACHED:
            return None
        raise CuptiError(
            f"cuptiActivityGetNextRecord failed with {self._result_string(rc)}"
        )

    def activity_get_num_dropped_records(self, ctx: int, stream_id: int) -> int:
        dropped = ctypes.c_size_t()
        rc = self._lib.cuptiActivityGetNumDroppedRecords(
            ctypes.c_void_p(ctx), ctypes.c_uint32(stream_id), ctypes.byref(dropped)
        )
        return dropped.value if rc == CUPTI_SUCCESS else 0

    def activity_register_callbacks(
        self, request_addr: int, complete_addr: int
    ) -> None:
        self._check(
            self._lib.cuptiActivityRegisterCallbacks(
                ctypes.c_void_p(request_addr), ctypes.c_void_p(complete_addr)
            ),
            "cuptiActivityRegisterCallbacks",
        )

    def activity_register_timestamp_callback(self, callback_addr: int) -> None:
        self._check(
            self._lib.cuptiActivityRegisterTimestampCallback(
                ctypes.c_void_p(callback_addr)
            ),
            "cuptiActivityRegisterTimestampCallback",
        )

    def activity_push_external_correlation_id(
        self, external_id: int, kind: ExternalCorrelationKind | None = None
    ) -> bool:
        """Push an external-correlation id (default kind CUSTOM1). Best-effort:
        returns False on failure."""
        if kind is None:
            kind = _cupti().ExternalCorrelationKind.CUSTOM1
        rc = self._lib.cuptiActivityPushExternalCorrelationId(
            kind, ctypes.c_uint64(external_id)
        )
        return rc == CUPTI_SUCCESS

    def activity_pop_external_correlation_id(
        self, kind: ExternalCorrelationKind | None = None
    ) -> int | None:
        """Pop the most recent external-correlation id (default kind CUSTOM1), or
        None on failure."""
        if kind is None:
            kind = _cupti().ExternalCorrelationKind.CUSTOM1
        last = ctypes.c_uint64()
        rc = self._lib.cuptiActivityPopExternalCorrelationId(kind, ctypes.byref(last))
        return last.value if rc == CUPTI_SUCCESS else None

    def activity_enable_hw_trace(self, enabled: bool) -> None:
        self._check(
            self._lib.cuptiActivityEnableHWTrace(1 if enabled else 0),
            "cuptiActivityEnableHWTrace",
        )

    def read_activity_kind(self, record_addr: int) -> int:
        """The leading CUpti_ActivityKind int at the head of an activity record."""
        return ctypes.c_int.from_address(record_addr).value

    # --- v2 / user-defined-records -----------------------------------------

    def has_v2(self) -> bool:
        """True if libcupti exposes the v2 user-defined-record API (>= 13.2)."""
        return self.get_version() >= _MIN_V2_VERSION and hasattr(
            self._lib, "cuptiActivityEnable_v2"
        )

    def subscribe(self, allow_multiple: bool = True) -> int:
        """cuptiSubscribe_v2 with a no-op callback -> opaque subscriber handle
        (the v2 activity API is subscription-scoped). ``allow_multiple`` requests
        coexistence with another CUPTI subscriber (e.g. Kineto); CUPTI returns
        CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED if it can't be honored."""
        sub = ctypes.c_void_p()
        params = _SubscriberParams(
            structSize=ctypes.sizeof(_SubscriberParams),
            subscriberName=b"torch-cupti-monitor",
            oldSubscriberName=None,
            oldSubscriberSize=0,
            allowMultipleSubscribers=1 if allow_multiple else 0,
        )
        self._check(
            self._lib.cuptiSubscribe_v2(
                ctypes.byref(sub), _NOOP_CB, None, ctypes.byref(params)
            ),
            "cuptiSubscribe_v2",
        )
        if sub.value is None:
            raise CuptiError("cuptiSubscribe returned a null subscriber handle")
        return sub.value

    def unsubscribe(self, sub_handle: int) -> None:
        self._check(
            self._lib.cuptiUnsubscribe(ctypes.c_void_p(sub_handle)),
            "cuptiUnsubscribe",
        )

    def arm_user_defined_records(
        self, sub_handle: int, request_addr: int, complete_addr: int
    ) -> None:
        """Turn on user-defined records for the subscription and register the v2
        buffer callbacks (the native pool's version=2 request/complete)."""
        enabled = ctypes.c_uint8(1)
        size = ctypes.c_size_t(1)
        self._check(
            self._lib.cuptiActivitySetAttribute_v2(
                ctypes.c_void_p(sub_handle),
                _ATTR_USER_DEFINED_RECORDS,
                ctypes.byref(size),
                ctypes.byref(enabled),
            ),
            "cuptiActivitySetAttribute_v2",
        )
        self._check(
            self._lib.cuptiActivityRegisterCallbacks_v2(
                ctypes.c_void_p(sub_handle),
                ctypes.c_void_p(request_addr),
                ctypes.c_void_p(complete_addr),
            ),
            "cuptiActivityRegisterCallbacks_v2",
        )

    def activity_enable_v2(
        self, sub_handle: int, kind: ActivityKind, field_ids: Iterable[int]
    ) -> None:
        """Enable a kind with a user-defined field selection. CUPTI requires the
        FIELD_KIND id (0) to be the first selected field."""
        ordered = (0, *sorted(f for f in field_ids if f != 0))
        arr = (ctypes.c_int * len(ordered))(*ordered)
        sel = _UDFieldSelection(
            structSize=ctypes.sizeof(_UDFieldSelection),
            numFields=len(ordered),
            pFieldIds=ctypes.cast(arr, ctypes.POINTER(ctypes.c_int)),
        )
        cfg = _UDActivityConfig(
            structSize=ctypes.sizeof(_UDActivityConfig), fieldSelection=sel
        )
        self._check(
            self._lib.cuptiActivityEnable_v2(
                ctypes.c_void_p(sub_handle), kind, ctypes.byref(cfg)
            ),
            "cuptiActivityEnable_v2",
        )

    def activity_disable_v2(self, sub_handle: int, kind: ActivityKind) -> None:
        self._check(
            self._lib.cuptiActivityDisable_v2(ctypes.c_void_p(sub_handle), kind),
            "cuptiActivityDisable_v2",
        )

    def activity_push_external_correlation_id_v2(
        self,
        sub_handle: int,
        external_id: int,
        kind: ExternalCorrelationKind | None = None,
    ) -> bool:
        if kind is None:
            kind = _cupti().ExternalCorrelationKind.CUSTOM1
        rc = self._lib.cuptiActivityPushExternalCorrelationId_v2(
            ctypes.c_void_p(sub_handle), kind, ctypes.c_uint64(external_id)
        )
        return rc == CUPTI_SUCCESS

    def activity_pop_external_correlation_id_v2(
        self, sub_handle: int, kind: ExternalCorrelationKind | None = None
    ) -> int | None:
        if kind is None:
            kind = _cupti().ExternalCorrelationKind.CUSTOM1
        last = ctypes.c_uint64()
        rc = self._lib.cuptiActivityPopExternalCorrelationId_v2(
            ctypes.c_void_p(sub_handle), kind, ctypes.byref(last)
        )
        return last.value if rc == CUPTI_SUCCESS else None


@lru_cache(maxsize=1)
def pylibcupti() -> _PyLibCupti:
    """The process-wide CUPTI Activity API wrapper: libcupti loaded and ctypes
    prototypes bound once. All libcupti calls go through this object -- callers
    never touch the CDLL or ctypes directly."""
    lib = ctypes.CDLL(find_cupti_library())
    _configure_ctypes(lib)
    return _PyLibCupti(lib)
