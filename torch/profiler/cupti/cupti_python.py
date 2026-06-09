# mypy: allow-untyped-defs
"""CUPTI type and constant definitions used by the in-process monitor.

This module is the single place that reaches into the ``cupti-python`` package
and into CUPTI's ABI constants. The cupti-python module is imported lazily and
cached by :func:`_cupti`, and the activity-kind enum the monitor needs is
re-exported as a module attribute resolved on first access (see
:func:`__getattr__`). Callers therefore just import what they use --
``from .cupti_python import ActivityKind`` -- and write ``ActivityKind.MEMCPY``
directly (cupti-python's enums are ``IntEnum``, so kind members compare equal to
the raw integer kind read from a record), while the cupti import stays deferred
until one of those names is first touched.
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
        ExternalCorrelationKind,
    )


# Environment override for the libcupti to dlopen; see find_cupti_library().
LIBCUPTI_PATH_ENV = "TORCH_CUPTI_MONITOR_LIBCUPTI_PATH"

# CUPTI C-API result/flag constants (cupti_result.h / cupti_activity.h). These
# are stable ABI values, so they are spelled out rather than resolved.
CUPTI_SUCCESS = 0
CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1

# CUpti_ActivityAttribute::CUPTI_ACTIVITY_ATTR_USER_DEFINED_RECORDS (not surfaced
# by cupti-python); set on the subscription to turn on the v2 user-defined-record
# path.
_ATTR_USER_DEFINED_RECORDS = 11

# Minimum libcupti the monitor supports. The v2 user-defined-record API arrived in
# 13.2, but only 13.3 populates pBufferCompleteInfo->ppRecordLayouts (CUPTI's own
# per-kind record layout) that the monitor decodes against, so 13.3 is the floor.
LIBCUPTI_MIN_VERSION = 130300

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

# cupti-python names re-exported lazily by __getattr__: the enums the monitor and
# trace builder need. Resolved off the cupti module on first access.
_REEXPORTED = frozenset(
    {
        "ActivityKind",
        "ExternalCorrelationKind",
        "Runtime_api_trace_cbid",
        "Driver_api_trace_cbid",
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
    lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
    lib.cuptiActivityFlushAll.restype = ctypes.c_int
    lib.cuptiActivityGetNumDroppedRecords.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cuptiActivityGetNumDroppedRecords.restype = ctypes.c_int
    lib.cuptiActivityEnableHWTrace.argtypes = [ctypes.c_uint8]
    lib.cuptiActivityEnableHWTrace.restype = ctypes.c_int
    lib.cuptiGetResultString.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.cuptiGetResultString.restype = ctypes.c_int
    lib.cuptiFinalize.argtypes = []
    lib.cuptiFinalize.restype = ctypes.c_int

    # User-defined-record (subscription) API -- present in libcupti >= 13.2; guarded
    # so configuring against an older libcupti still succeeds (the monitor's
    # LIBCUPTI_MIN_VERSION check then fails fast at start).
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
    if hasattr(lib, "cuptiGetTimestamp_v2"):
        lib.cuptiGetTimestamp_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.POINTER(ctypes.c_uint64),
        ]
        lib.cuptiGetTimestamp_v2.restype = ctypes.c_int
    # External correlation push/pop. The plain (v1) calls return
    # CUPTI_ERROR_NOT_COMPATIBLE while a user-defined-record subscriber is active
    # (same as cuptiGetTimestamp), so the subscriber-aware _v2 variants are required
    # on the v2 path; bind them when present (libcupti >= 13.3) and fall back to v1.
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
    if hasattr(lib, "cuptiActivityPushExternalCorrelationId_v2"):
        lib.cuptiActivityPushExternalCorrelationId_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        lib.cuptiActivityPushExternalCorrelationId_v2.restype = ctypes.c_int
    if hasattr(lib, "cuptiActivityPopExternalCorrelationId_v2"):
        lib.cuptiActivityPopExternalCorrelationId_v2.argtypes = [
            ctypes.c_void_p,  # CUpti_SubscriberHandle subscriber
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

    def get_timestamp(self, sub_handle: int) -> int:
        """CUPTI's normalized nanosecond clock for a subscriber -- the same timebase
        as activity record START/END timestamps, so a value captured here is directly
        comparable to decoded record timestamps. The subscriber-aware _v2 form is
        required while a user-defined-record subscriber is active: plain
        cuptiGetTimestamp returns CUPTI_ERROR_NOT_COMPATIBLE under UDR on libcupti
        13.3 (it works only with no active v2 subscriber)."""
        ts = ctypes.c_uint64()
        self._check(
            self._lib.cuptiGetTimestamp_v2(
                ctypes.c_void_p(sub_handle), ctypes.byref(ts)
            ),
            "cuptiGetTimestamp_v2",
        )
        return ts.value

    def activity_flush_all(self, forced: bool) -> None:
        flag = CUPTI_ACTIVITY_FLAG_FLUSH_FORCED if forced else 0
        self._check(self._lib.cuptiActivityFlushAll(flag), "cuptiActivityFlushAll")

    def activity_get_num_dropped_records(self, ctx: int, stream_id: int) -> int:
        dropped = ctypes.c_size_t()
        rc = self._lib.cuptiActivityGetNumDroppedRecords(
            ctypes.c_void_p(ctx), ctypes.c_uint32(stream_id), ctypes.byref(dropped)
        )
        return dropped.value if rc == CUPTI_SUCCESS else 0

    def activity_enable_hw_trace(self, enabled: bool) -> None:
        self._check(
            self._lib.cuptiActivityEnableHWTrace(1 if enabled else 0),
            "cuptiActivityEnableHWTrace",
        )

    def finalize(self) -> None:
        """cuptiFinalize -- detach and release ALL of CUPTI process-wide. This is a
        global, heavy reset for explicit *synchronous* teardown: e.g. releasing a
        stock Kineto session's CUPTI subscriber before a CUPTI-monitor session
        subscribes. The monitor itself never calls this at stop() -- it disarms
        user-defined records + unsubscribes instead, because cuptiFinalize is global
        (would clobber a concurrent consumer) and, run asynchronously (Kineto's
        TEARDOWN_CUPTI), can deadlock against another thread's CUPTI calls."""
        self._check(self._lib.cuptiFinalize(), "cuptiFinalize")

    # --- user-defined-records (subscription API) ---------------------------

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

    def disarm_user_defined_records(self, sub_handle: int) -> None:
        """Turn user-defined records back off for the subscription (the inverse of
        arm_user_defined_records' set-attribute). UDR mode changes how CUPTI lays
        out activity records, so leaving it on can leave a following classic
        consumer (e.g. Kineto) unable to decode -- reset it before unsubscribing."""
        disabled = ctypes.c_uint8(0)
        size = ctypes.c_size_t(1)
        self._check(
            self._lib.cuptiActivitySetAttribute_v2(
                ctypes.c_void_p(sub_handle),
                _ATTR_USER_DEFINED_RECORDS,
                ctypes.byref(size),
                ctypes.byref(disabled),
            ),
            "cuptiActivitySetAttribute_v2",
        )

    def activity_enable(
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

    def activity_disable(self, sub_handle: int, kind: ActivityKind) -> None:
        self._check(
            self._lib.cuptiActivityDisable_v2(ctypes.c_void_p(sub_handle), kind),
            "cuptiActivityDisable_v2",
        )

    def activity_push_external_correlation_id(
        self,
        external_id: int,
        kind: ExternalCorrelationKind | None = None,
        sub_handle: int | None = None,
    ) -> bool:
        """Push an external-correlation id (default kind CUSTOM1) onto CUPTI's
        process-global stack. Best-effort: returns False on failure. Pass
        ``sub_handle`` on the v2 path -- the plain call returns NOT_COMPATIBLE while
        a user-defined-record subscriber is active, so the subscriber-aware _v2
        variant is used when a handle is given and it's available."""
        if kind is None:
            kind = _cupti().ExternalCorrelationKind.CUSTOM1
        if sub_handle is not None and hasattr(
            self._lib, "cuptiActivityPushExternalCorrelationId_v2"
        ):
            rc = self._lib.cuptiActivityPushExternalCorrelationId_v2(
                sub_handle, int(kind), ctypes.c_uint64(external_id)
            )
        else:
            rc = self._lib.cuptiActivityPushExternalCorrelationId(
                int(kind), ctypes.c_uint64(external_id)
            )
        return rc == CUPTI_SUCCESS

    def activity_pop_external_correlation_id(
        self,
        kind: ExternalCorrelationKind | None = None,
        sub_handle: int | None = None,
    ) -> int | None:
        """Pop the most recent external-correlation id (default kind CUSTOM1), or
        None on failure. Pass ``sub_handle`` on the v2 path (see the push docstring)."""
        if kind is None:
            kind = _cupti().ExternalCorrelationKind.CUSTOM1
        last = ctypes.c_uint64()
        if sub_handle is not None and hasattr(
            self._lib, "cuptiActivityPopExternalCorrelationId_v2"
        ):
            rc = self._lib.cuptiActivityPopExternalCorrelationId_v2(
                sub_handle, int(kind), ctypes.byref(last)
            )
        else:
            rc = self._lib.cuptiActivityPopExternalCorrelationId(
                int(kind), ctypes.byref(last)
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
