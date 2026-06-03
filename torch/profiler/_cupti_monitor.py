# mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import ctypes
import json
import os
import queue
import struct
import subprocess
import threading
import time
from collections.abc import Callable, Iterable  # noqa: TC003
from pathlib import Path
from typing import Any, cast

import torch


_PY_PROFILER = torch._C._profiler


_LIBCUPTI_PATH_ENV = "TORCH_CUPTI_MONITOR_LIBCUPTI_PATH"

_CUPTI_SUCCESS = 0
_CUPTI_ERROR_MAX_LIMIT_REACHED = 12

_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1

_DEFAULT_BUFFER_SIZE = 4 * 1024 * 1024
_DEFAULT_FLUSH_PERIOD_S = 1.0

_META_FILE = "cupti_monitor_meta.bin"
_RAW_BUFFER_FILE = "cupti_monitor_raw_buffers.bin"

_RAW_CHUNK_MAGIC = b"CUPMRBUF"
_RAW_CHUNK_VERSION = 2
_RAW_CHUNK_HEADER = struct.Struct("<8sIIIIQQQ")
_META_MAGIC = b"CUPMMETA"
_META_VERSION = 2
_META_HEADER = struct.Struct("<8sII")

_RECORD_KIND_KERNEL = 1
_RECORD_KIND_MEMCPY = 2
_RECORD_KIND_MEMSET = 3

_OVERHEAD_KIND_NAMES = {
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
_DEMANGLE_CACHE: dict[str, str] = {}
_cc = None
_USER_EXTERNAL_CORRELATION_KIND: int | None = None
_CUPTI_ACTIVITY_KIND_MEMCPY: int | None = None
_CUPTI_ACTIVITY_KIND_MEMSET: int | None = None
_CUPTI_ACTIVITY_KIND_DRIVER: int | None = None
_CUPTI_ACTIVITY_KIND_RUNTIME: int | None = None
_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: int | None = None
_CUPTI_ACTIVITY_KIND_OVERHEAD: int | None = None
_CUPTI_ACTIVITY_KIND_CUDA_EVENT: int | None = None
_CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: int | None = None
_CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: int | None = None
_DISABLED_RUNTIME_CBIDS: tuple[int, ...] = ()
_DISABLED_DRIVER_CBIDS: tuple[int, ...] = ()
_KERNEL_DTYPE = None
_MEMCPY_DTYPE = None
_MEMSET_DTYPE = None
_API_DTYPE = None
_EXTERNAL_CORRELATION_DTYPE = None
_OVERHEAD_DTYPE = None
_CUDA_EVENT_DTYPE = None
_SYNCHRONIZATION_DTYPE = None


def _find_cupti_library() -> str:
    override = os.environ.get(_LIBCUPTI_PATH_ENV)
    if override:
        return override
    from cuda.pathfinder import (  # pyrefly: ignore[missing-import]
        load_nvidia_dynamic_lib,
    )

    return load_nvidia_dynamic_lib("cupti").abs_path


def _require_cupti_python():
    global _cc
    global _USER_EXTERNAL_CORRELATION_KIND
    global _CUPTI_ACTIVITY_KIND_MEMCPY
    global _CUPTI_ACTIVITY_KIND_MEMSET
    global _CUPTI_ACTIVITY_KIND_DRIVER
    global _CUPTI_ACTIVITY_KIND_RUNTIME
    global _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
    global _CUPTI_ACTIVITY_KIND_OVERHEAD
    global _CUPTI_ACTIVITY_KIND_CUDA_EVENT
    global _CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
    global _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION
    global _DISABLED_RUNTIME_CBIDS
    global _DISABLED_DRIVER_CBIDS
    global _KERNEL_DTYPE
    global _MEMCPY_DTYPE
    global _MEMSET_DTYPE
    global _API_DTYPE
    global _EXTERNAL_CORRELATION_DTYPE
    global _OVERHEAD_DTYPE
    global _CUDA_EVENT_DTYPE
    global _SYNCHRONIZATION_DTYPE

    if _cc is not None:
        return _cc

    try:
        from cupti import cupti as imported_cc  # pyrefly: ignore[missing-import]
    except ModuleNotFoundError as exc:
        raise ImportError(
            "torch.profiler._cupti_monitor requires the cupti-python package. "
            "Install cupti-python to use the experimental CUPTI monitor."
        ) from exc

    _cc = imported_cc
    _USER_EXTERNAL_CORRELATION_KIND = int(_cc.ExternalCorrelationKind.CUSTOM1)
    _CUPTI_ACTIVITY_KIND_MEMCPY = int(_cc.ActivityKind.MEMCPY)
    _CUPTI_ACTIVITY_KIND_MEMSET = int(_cc.ActivityKind.MEMSET)
    _CUPTI_ACTIVITY_KIND_DRIVER = int(_cc.ActivityKind.DRIVER)
    _CUPTI_ACTIVITY_KIND_RUNTIME = int(_cc.ActivityKind.RUNTIME)
    _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = int(_cc.ActivityKind.CONCURRENT_KERNEL)
    _CUPTI_ACTIVITY_KIND_OVERHEAD = int(_cc.ActivityKind.OVERHEAD)
    _CUPTI_ACTIVITY_KIND_CUDA_EVENT = int(_cc.ActivityKind.CUDA_EVENT)
    _CUPTI_ACTIVITY_KIND_SYNCHRONIZATION = int(_cc.ActivityKind.SYNCHRONIZATION)
    _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = int(
        _cc.ActivityKind.EXTERNAL_CORRELATION
    )
    _DISABLED_RUNTIME_CBIDS = tuple(
        int(cbid)
        for cbid in (
            _cc.Runtime_api_trace_cbid.cudaGetDevice_v3020,
            _cc.Runtime_api_trace_cbid.cudaSetDevice_v3020,
            _cc.Runtime_api_trace_cbid.cudaGetLastError_v3020,
            _cc.Runtime_api_trace_cbid.cudaEventCreate_v3020,
            _cc.Runtime_api_trace_cbid.cudaEventCreateWithFlags_v3020,
            _cc.Runtime_api_trace_cbid.cudaEventDestroy_v3020,
        )
    )
    _DISABLED_DRIVER_CBIDS = tuple(
        int(cbid)
        for cbid in (
            _cc.Driver_api_trace_cbid.cuKernelGetAttribute,
            _cc.Driver_api_trace_cbid.cuDevicePrimaryCtxGetState,
            _cc.Driver_api_trace_cbid.cuCtxGetCurrent,
        )
    )
    _KERNEL_DTYPE = _cc.activity_kernel11_dtype
    _MEMCPY_DTYPE = _cc.activity_memcpy6_dtype
    _MEMSET_DTYPE = _cc.activity_memset4_dtype
    _API_DTYPE = _cc.activity_api_dtype
    _EXTERNAL_CORRELATION_DTYPE = _cc.activity_external_correlation_dtype
    _OVERHEAD_DTYPE = _cc.activity_overhead3_dtype
    _CUDA_EVENT_DTYPE = _cc.activity_cuda_event2_dtype
    _SYNCHRONIZATION_DTYPE = _cc.activity_synchronization2_dtype
    return _cc


def _default_always_on_activities() -> tuple[int, ...]:
    _require_cupti_python()
    return (
        int(cast(int, _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_MEMCPY)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_MEMSET)),
    )


def _default_trace_window_activities() -> tuple[int, ...]:
    _require_cupti_python()
    return (
        int(cast(int, _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_MEMCPY)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_MEMSET)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_RUNTIME)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_DRIVER)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_OVERHEAD)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_CUDA_EVENT)),
        int(cast(int, _CUPTI_ACTIVITY_KIND_SYNCHRONIZATION)),
    )


def _has_active_cuda_context() -> bool:
    try:
        from cuda.bindings import (  # pyrefly: ignore[missing-import]
            driver as cuda_driver,
        )
    except ImportError:
        return False
    rc, ctx = cuda_driver.cuCtxGetCurrent()
    if rc == cuda_driver.CUresult.CUDA_SUCCESS:
        return ctx is not None
    if rc == cuda_driver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        return False
    raise RuntimeError(f"cuCtxGetCurrent failed with rc={rc}")


def _current_thread_resource_tuple() -> tuple[int, int, int]:
    opaque_tid = ctypes.c_int32(threading.get_ident() & 0xFFFFFFFF).value
    return (os.getpid(), opaque_tid, threading.get_native_id())


def _cuda_version_string() -> str:
    return torch.version.cuda or ""


def _safe_json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _open_binary_append(path: Path):
    return open(path, "ab", buffering=0)


class _CuptiError(RuntimeError):
    pass


def _default_graph_annotation_resolver(
    graph_node_id: int, record_kind: int, correlation_id: int
) -> Any | None:
    del record_kind, correlation_id
    if graph_node_id == 0:
        return None
    try:
        from torch.cuda._graph_annotations import get_kernel_annotations

        annotations = get_kernel_annotations()
    except Exception:
        return None
    return annotations.get(graph_node_id)


def _decode_c_string(ptr: int | None, default: str) -> str:
    if not ptr:
        return default
    value = ctypes.cast(ptr, ctypes.c_char_p).value
    if value is None:
        return default
    return value.decode(errors="replace")


def _demangle_symbol(name: str) -> str:
    if not name.startswith("_Z"):
        return name
    cached = _DEMANGLE_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        proc = subprocess.run(
            ["/bin/c++filt", "-n", name],
            check=True,
            capture_output=True,
            text=True,
        )
        demangled = proc.stdout.strip() or name
    except Exception:
        demangled = name
    _DEMANGLE_CACHE[name] = demangled
    return demangled


class CuptiMonitor:
    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        activities: Iterable[int] | None = None,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        flush_period_s: float = _DEFAULT_FLUSH_PERIOD_S,
        annotation_resolver: Callable[[int, int, int], Any | None] | None = None,
    ) -> None:
        _require_cupti_python()
        self.output_dir = Path(output_dir)
        self.activities = tuple(activities or _default_always_on_activities())
        self.buffer_size = buffer_size
        self.flush_period_s = flush_period_s
        self.annotation_resolver = (
            annotation_resolver or _default_graph_annotation_resolver
        )

        self._lib = ctypes.CDLL(_find_cupti_library())
        self._setup_prototypes()

        self._lock = threading.Lock()
        self._processing_done = threading.Condition(self._lock)
        self._free_buffers: list[int] = []
        self._buffer_keepalive: dict[int, Any] = {}
        self._completed_queue: queue.SimpleQueue[tuple[int, int, int, int]] = (
            queue.SimpleQueue()
        )
        self._started = False
        self._callbacks_registered = False
        self._enabled_activities: set[int] = set()
        self._worker_stop = threading.Event()
        self._flush_stop = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._flush_thread: threading.Thread | None = None
        self._worker_error: BaseException | None = None
        self._trace_window_active = False
        self._trace_window_prepared = False
        self._trace_window_events: list[dict[str, Any]] = []
        self._trace_window_extra_activities: tuple[int, ...] = ()
        self._trace_window_start_ns = 0
        self._trace_window_user_annotations: dict[int, str] = {}
        self._next_user_external_id = 1
        self._thread_resource_map: dict[int, dict[int, int]] = {}
        self._processing_inflight = 0
        self._time_converter = None
        self._timestamp_callback = None
        self._session_start_unix_ns = 0
        self._session_start_approx_ns = 0
        self._session_start_calibrated_unix_ns = 0

        self._raw_buffers_fp = None

        self._buffers_requested = 0
        self._buffers_completed = 0
        self._valid_bytes = 0
        self._max_outstanding = 0
        self._dropped_records = 0
        self._raw_chunk_count = 0

        req_type = ctypes.CFUNCTYPE(
            None,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        comp_type = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
        )

        @req_type
        def request_cb(buffer_ptr, size_ptr, max_records_ptr):
            with self._lock:
                if self._free_buffers:
                    ptr = self._free_buffers.pop()
                else:
                    arr = (ctypes.c_uint8 * self.buffer_size)()
                    ptr = ctypes.addressof(arr)
                    self._buffer_keepalive[ptr] = arr
                self._buffers_requested += 1
                outstanding = self._buffers_requested - self._buffers_completed
                self._max_outstanding = max(self._max_outstanding, outstanding)

            buffer_ptr[0] = ptr
            size_ptr[0] = self.buffer_size
            max_records_ptr[0] = 0

        @comp_type
        def complete_cb(ctx, stream_id, buffer, size, valid_size):
            with self._lock:
                self._buffers_completed += 1
                self._valid_bytes += int(valid_size)
            self._completed_queue.put(
                (
                    int(ctx) if ctx else 0,
                    int(stream_id),
                    int(buffer),
                    int(valid_size),
                )
            )

        self._request_cb = request_cb
        self._complete_cb = complete_cb

    def _setup_prototypes(self) -> None:
        self._lib.cuptiGetVersion.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
        self._lib.cuptiGetVersion.restype = ctypes.c_int
        self._lib.cuptiActivityRegisterCallbacks.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        self._lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
        self._lib.cuptiActivityEnable.restype = ctypes.c_int
        self._lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
        self._lib.cuptiActivityDisable.restype = ctypes.c_int
        if hasattr(self._lib, "cuptiActivityEnableRuntimeApi"):
            self._lib.cuptiActivityEnableRuntimeApi.argtypes = [
                ctypes.c_uint32,
                ctypes.c_uint8,
            ]
            self._lib.cuptiActivityEnableRuntimeApi.restype = ctypes.c_int
        if hasattr(self._lib, "cuptiActivityEnableDriverApi"):
            self._lib.cuptiActivityEnableDriverApi.argtypes = [
                ctypes.c_uint32,
                ctypes.c_uint8,
            ]
            self._lib.cuptiActivityEnableDriverApi.restype = ctypes.c_int
        self._lib.cuptiActivityFlushAll.argtypes = [ctypes.c_uint32]
        self._lib.cuptiActivityFlushAll.restype = ctypes.c_int
        self._lib.cuptiActivityGetNextRecord.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._lib.cuptiActivityGetNextRecord.restype = ctypes.c_int
        self._lib.cuptiActivityGetNumDroppedRecords.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self._lib.cuptiActivityGetNumDroppedRecords.restype = ctypes.c_int
        self._lib.cuptiActivityEnableHWTrace.argtypes = [ctypes.c_uint8]
        self._lib.cuptiActivityEnableHWTrace.restype = ctypes.c_int
        self._lib.cuptiActivityRegisterTimestampCallback.argtypes = [ctypes.c_void_p]
        self._lib.cuptiActivityRegisterTimestampCallback.restype = ctypes.c_int
        self._lib.cuptiActivityPushExternalCorrelationId.argtypes = [
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        self._lib.cuptiActivityPushExternalCorrelationId.restype = ctypes.c_int
        self._lib.cuptiActivityPopExternalCorrelationId.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._lib.cuptiActivityPopExternalCorrelationId.restype = ctypes.c_int
        self._lib.cuptiGetResultString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._lib.cuptiGetResultString.restype = ctypes.c_int

    def _cupti_version(self) -> int:
        version = ctypes.c_uint32()
        self._check(
            self._lib.cuptiGetVersion(ctypes.byref(version)),
            "cuptiGetVersion",
        )
        return int(version.value)

    def _result_string(self, rc: int) -> str:
        result = ctypes.c_char_p()
        rc2 = self._lib.cuptiGetResultString(rc, ctypes.byref(result))
        if rc2 == _CUPTI_SUCCESS and result.value is not None:
            return result.value.decode()
        return f"rc={rc}"

    def _check(self, rc: int, name: str) -> None:
        if rc != _CUPTI_SUCCESS:
            raise _CuptiError(f"{name} failed with {self._result_string(rc)}")

    def register_callbacks(self) -> None:
        if self._callbacks_registered:
            return
        self._check(
            self._lib.cuptiActivityRegisterCallbacks(
                ctypes.cast(self._request_cb, ctypes.c_void_p),
                ctypes.cast(self._complete_cb, ctypes.c_void_p),
            ),
            "cuptiActivityRegisterCallbacks",
        )
        self._callbacks_registered = True

    def register_timestamp_callback(self) -> None:
        callback_addr = getattr(  # noqa: B009
            _PY_PROFILER, "_cupti_approximate_time_callback_address"
        )()
        self._check(
            self._lib.cuptiActivityRegisterTimestampCallback(
                ctypes.c_void_p(callback_addr)
            ),
            "cuptiActivityRegisterTimestampCallback",
        )
        self._timestamp_callback = callback_addr

    def start(self) -> None:
        if self._started:
            raise RuntimeError("CUPTI monitor is already started")
        self.register_callbacks()
        self._time_converter = getattr(  # noqa: B009
            _PY_PROFILER, "_ApproximateClockToUnixTimeConverter"
        )()
        self.register_timestamp_callback()
        self._session_start_unix_ns = time.time_ns()
        self._session_start_approx_ns = int(
            getattr(_PY_PROFILER, "_get_approximate_time")()  # noqa: B009
        )
        self._session_start_calibrated_unix_ns = self._convert_time(
            self._session_start_approx_ns
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_outputs()
        self._write_meta_file()
        self._worker_stop.clear()
        self._flush_stop.clear()
        self._worker_error = None
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="torch-cupti-monitor-worker",
            daemon=True,
        )
        self._worker_thread.start()
        if self.flush_period_s > 0:
            self._flush_thread = threading.Thread(
                target=self._flush_loop,
                name="torch-cupti-monitor-flush",
                daemon=True,
            )
            self._flush_thread.start()
        self._record_current_thread_info()
        self.enable_activities(self.activities)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._check(
            self._lib.cuptiActivityFlushAll(_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),
            "cuptiActivityFlushAll",
        )
        for activity in reversed(tuple(self._enabled_activities)):
            self._check(
                self._lib.cuptiActivityDisable(activity),
                "cuptiActivityDisable",
            )
        self._enabled_activities.clear()
        self._started = False
        self._flush_stop.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        self._worker_stop.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None
        if self._worker_error is not None:
            raise RuntimeError("CUPTI monitor worker failed") from self._worker_error
        self._close_outputs()
        self._time_converter = None
        self._timestamp_callback = None

    def flush(self, *, forced: bool = False) -> None:
        flag = _CUPTI_ACTIVITY_FLAG_FLUSH_FORCED if forced else 0
        self._check(self._lib.cuptiActivityFlushAll(flag), "cuptiActivityFlushAll")

    def _convert_time(self, value: int) -> int:
        if value == 0:
            return 0
        if self._time_converter is None:
            return value
        return int(self._time_converter.to_unix_ns(int(value)))

    def enable_activities(self, activities: Iterable[int]) -> tuple[int, ...]:
        newly_enabled = []
        for activity in activities:
            if activity in self._enabled_activities:
                continue
            self._check(self._lib.cuptiActivityEnable(activity), "cuptiActivityEnable")
            self._apply_activity_filters(activity)
            self._enabled_activities.add(activity)
            newly_enabled.append(activity)
        return tuple(newly_enabled)

    def disable_activities(self, activities: Iterable[int]) -> None:
        for activity in activities:
            if activity not in self._enabled_activities:
                continue
            self._check(
                self._lib.cuptiActivityDisable(activity),
                "cuptiActivityDisable",
            )
            self._enabled_activities.remove(activity)

    def _apply_activity_filters(self, activity: int) -> None:
        if activity == _CUPTI_ACTIVITY_KIND_RUNTIME and hasattr(
            self._lib, "cuptiActivityEnableRuntimeApi"
        ):
            for cbid in _DISABLED_RUNTIME_CBIDS:
                self._check(
                    self._lib.cuptiActivityEnableRuntimeApi(cbid, 0),
                    "cuptiActivityEnableRuntimeApi",
                )
        if activity == _CUPTI_ACTIVITY_KIND_DRIVER and hasattr(
            self._lib, "cuptiActivityEnableDriverApi"
        ):
            for cbid in _DISABLED_DRIVER_CBIDS:
                self._check(
                    self._lib.cuptiActivityEnableDriverApi(cbid, 0),
                    "cuptiActivityEnableDriverApi",
                )

    def prepare_trace_window(
        self, activities: Iterable[int] | None = None
    ) -> tuple[int, ...]:
        if not self._started:
            raise RuntimeError(
                "CUPTI monitor must be started before prepare_trace_window"
            )
        if self._trace_window_prepared:
            raise RuntimeError("A trace window is already prepared")
        activities = tuple(activities or _default_trace_window_activities())
        self._record_current_thread_info()
        newly_enabled = self.enable_activities(activities)
        with self._lock:
            self._trace_window_events = []
            self._trace_window_prepared = True
            self._trace_window_active = False
            self._trace_window_extra_activities = newly_enabled
            self._trace_window_start_ns = 0
            self._trace_window_user_annotations = {}
        return newly_enabled

    def start_trace_window(self) -> None:
        if not self._started:
            raise RuntimeError(
                "CUPTI monitor must be started before start_trace_window"
            )
        if not self._trace_window_prepared:
            raise RuntimeError("No prepared trace window")
        if self._trace_window_active:
            raise RuntimeError("A trace window is already active")
        self._record_current_thread_info()
        self.flush(forced=False)
        self._wait_for_processing_idle(timeout_s=5.0)
        with self._lock:
            self._trace_window_events = []
            self._trace_window_start_ns = self._convert_time(
                int(getattr(_PY_PROFILER, "_get_approximate_time")())  # noqa: B009
            )
            self._trace_window_active = True

    def begin_trace_window(
        self, activities: Iterable[int] | None = None
    ) -> tuple[int, ...]:
        newly_enabled = self.prepare_trace_window(activities)
        self.start_trace_window()
        return newly_enabled

    def end_trace_window(self) -> dict[str, Any]:
        if not self._trace_window_prepared:
            raise RuntimeError("No prepared trace window")
        self._record_current_thread_info()
        with self._lock:
            extra = self._trace_window_extra_activities
            start_ns = self._trace_window_start_ns
            user_annotations = dict(self._trace_window_user_annotations)
            thread_resource_map = {
                pid: dict(mapping) for pid, mapping in self._thread_resource_map.items()
            }
        self.disable_activities(extra)
        self.flush(forced=True)
        self._wait_for_processing_idle(timeout_s=5.0)
        with self._lock:
            events = list(self._trace_window_events)
            self._trace_window_events = []
            self._trace_window_prepared = False
            self._trace_window_active = False
            self._trace_window_extra_activities = ()
            self._trace_window_start_ns = 0
            self._trace_window_user_annotations = {}
        self.disable_activities(extra)
        return {
            "events": self._filter_trace_window_events(events, start_ns),
            "extra_activities": list(extra),
            "user_annotations": user_annotations,
            "thread_resource_map": thread_resource_map,
            "start_ns": start_ns,
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self._started,
                "activities": list(self._enabled_activities),
                "buffers_requested": self._buffers_requested,
                "buffers_completed": self._buffers_completed,
                "buffers_allocated": len(self._buffer_keepalive),
                "buffers_free": len(self._free_buffers),
                "max_outstanding_buffers": self._max_outstanding,
                "valid_total_mb": self._valid_bytes / (1024 * 1024),
                "dropped_records": self._dropped_records,
                "raw_chunks_written": self._raw_chunk_count,
                "output_dir": str(self.output_dir),
                "trace_window_prepared": self._trace_window_prepared,
                "trace_window_active": self._trace_window_active,
                "trace_window_start_ns": self._trace_window_start_ns,
            }

    def push_user_annotation(self, name: str) -> int | None:
        self._record_current_thread_info()
        with self._lock:
            if (
                not self._started
                or not self._trace_window_prepared
                or _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION
                not in self._enabled_activities
            ):
                return None
            external_id = self._next_user_external_id
            self._next_user_external_id += 1
            self._trace_window_user_annotations[external_id] = name
        self._check(
            self._lib.cuptiActivityPushExternalCorrelationId(
                _USER_EXTERNAL_CORRELATION_KIND, ctypes.c_uint64(external_id)
            ),
            "cuptiActivityPushExternalCorrelationId",
        )
        return external_id

    def pop_user_annotation(self) -> int | None:
        self._record_current_thread_info()
        with self._lock:
            if (
                not self._started
                or not self._trace_window_prepared
                or _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION
                not in self._enabled_activities
            ):
                return None
        last_id = ctypes.c_uint64()
        self._check(
            self._lib.cuptiActivityPopExternalCorrelationId(
                _USER_EXTERNAL_CORRELATION_KIND, ctypes.byref(last_id)
            ),
            "cuptiActivityPopExternalCorrelationId",
        )
        return int(last_id.value)

    def _open_outputs(self) -> None:
        self._raw_buffers_fp = _open_binary_append(self.output_dir / _RAW_BUFFER_FILE)

    def _close_outputs(self) -> None:
        if self._raw_buffers_fp is not None:
            self._raw_buffers_fp.close()
            self._raw_buffers_fp = None

    def _write_meta_file(self) -> None:
        meta = {
            "meta_version": _META_VERSION,
            "cupti_version": self._cupti_version(),
            "cuda_version": _cuda_version_string(),
            "hes_enabled": bool(_hes_enabled),
            "timestamp_mode": "approximate_clock",
            "session_start_unix_ns": self._session_start_unix_ns,
            "session_start_approx_ns": self._session_start_approx_ns,
            "session_start_calibrated_unix_ns": self._session_start_calibrated_unix_ns,
            "buffer_size": self.buffer_size,
            "flush_period_ns": int(self.flush_period_s * 1e9),
            "raw_buffer_dump": True,
            "activities": list(self.activities),
            "libcupti_path": _find_cupti_library(),
        }
        payload = _safe_json_dumps(meta)
        with open(self.output_dir / _META_FILE, "wb") as fp:
            fp.write(_META_HEADER.pack(_META_MAGIC, _META_VERSION, len(payload)))
            fp.write(payload)

    def _record_current_thread_info(self) -> None:
        pid, opaque_tid, sys_tid = _current_thread_resource_tuple()
        with self._lock:
            self._thread_resource_map.setdefault(pid, {})[opaque_tid] = sys_tid

    def _flush_loop(self) -> None:
        try:
            while not self._flush_stop.wait(self.flush_period_s):
                if self._started:
                    self.flush(forced=False)
        except BaseException as exc:
            self._worker_error = exc
            self._worker_stop.set()

    def _worker_loop(self) -> None:
        try:
            while True:
                if self._worker_stop.is_set():
                    try:
                        item = self._completed_queue.get_nowait()
                    except queue.Empty:
                        break
                else:
                    try:
                        item = self._completed_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                ctx, stream_id, buffer_ptr, valid_size = item
                with self._lock:
                    self._processing_inflight += 1
                try:
                    self._process_completed_buffer(
                        ctx, stream_id, buffer_ptr, valid_size
                    )
                finally:
                    with self._processing_done:
                        self._processing_inflight -= 1
                        self._processing_done.notify_all()
        except BaseException as exc:
            self._worker_error = exc
            self._worker_stop.set()

    def _wait_for_processing_idle(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        with self._processing_done:
            while True:
                if self._completed_queue.empty() and self._processing_inflight == 0:
                    return
                remaining = deadline - time.time()
                if remaining <= 0:
                    return
                self._processing_done.wait(timeout=min(0.05, remaining))

    def _process_completed_buffer(
        self, ctx: int, stream_id: int, buffer_ptr: int, valid_size: int
    ) -> None:
        if not self._trace_window_prepared:
            dropped = ctypes.c_size_t()
            rc = self._lib.cuptiActivityGetNumDroppedRecords(
                ctypes.c_void_p(ctx), ctypes.c_uint32(stream_id), ctypes.byref(dropped)
            )
            if rc == _CUPTI_SUCCESS:
                self._dropped_records += int(dropped.value)
            self._write_raw_buffer(ctx, stream_id, buffer_ptr, valid_size)
            with self._lock:
                self._free_buffers.append(buffer_ptr)
            return

        record_ptr = ctypes.c_void_p()
        while True:
            rc = self._lib.cuptiActivityGetNextRecord(
                ctypes.c_void_p(buffer_ptr),
                ctypes.c_size_t(valid_size),
                ctypes.byref(record_ptr),
            )
            if rc == _CUPTI_SUCCESS:
                record_addr = record_ptr.value
                if record_addr is None:
                    raise RuntimeError("CUPTI returned null activity record pointer")
                trace_event = self._decode_record(record_addr)
                if trace_event is not None:
                    with self._lock:
                        if self._trace_window_active:
                            self._trace_window_events.append(trace_event)
                continue
            if rc == _CUPTI_ERROR_MAX_LIMIT_REACHED:
                break
            raise _CuptiError(
                f"cuptiActivityGetNextRecord failed with {self._result_string(rc)}"
            )

        dropped = ctypes.c_size_t()
        rc = self._lib.cuptiActivityGetNumDroppedRecords(
            ctypes.c_void_p(ctx), ctypes.c_uint32(stream_id), ctypes.byref(dropped)
        )
        if rc == _CUPTI_SUCCESS:
            self._dropped_records += int(dropped.value)

        with self._lock:
            self._free_buffers.append(buffer_ptr)

    def _filter_trace_window_events(
        self, events: list[dict[str, Any]], start_ns: int
    ) -> list[dict[str, Any]]:
        if start_ns == 0:
            return events

        retained_correlations: set[int] = set()
        retained_events: list[dict[str, Any]] = []
        pending_gpu_events: list[dict[str, Any]] = []
        pending_external_events: list[dict[str, Any]] = []

        for event in events:
            kind = event.get("kind")
            if kind in {"cuda_runtime", "cuda_driver"}:
                event_start_ns = int(event.get("start_ns", 0))
                if event_start_ns >= start_ns:
                    retained_events.append(event)
                    correlation_id = int(event.get("correlation_id", 0))
                    if correlation_id != 0:
                        retained_correlations.add(correlation_id)
                continue

            if kind in {"kernel", "gpu_memcpy", "gpu_memset"}:
                pending_gpu_events.append(event)
                continue

            if kind == "external_correlation":
                pending_external_events.append(event)
                continue

            event_start_ns = int(event.get("start_ns", 0))
            if event_start_ns == 0 or event_start_ns >= start_ns:
                retained_events.append(event)

        for event in pending_external_events:
            correlation_id = int(event.get("correlation_id", 0))
            if correlation_id in retained_correlations:
                retained_events.append(event)

        for event in pending_gpu_events:
            correlation_id = int(event.get("correlation_id", 0))
            event_start_ns = int(event.get("start_ns", 0))
            if correlation_id in retained_correlations or (
                correlation_id == 0 and event_start_ns >= start_ns
            ):
                retained_events.append(event)

        return retained_events

    def _decode_record(self, record_addr: int) -> dict[str, Any] | None:
        cc = _require_cupti_python()
        kind = int(ctypes.c_int.from_address(record_addr).value)
        if kind == _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            record = cc.ActivityKernel11.from_ptr(record_addr, readonly=True)
            annotation = self.annotation_resolver(
                int(record.graph_node_id),
                _RECORD_KIND_KERNEL,
                int(record.correlation_id),
            )
            kernel_name = _demangle_symbol(record.name)
            start_ns = self._convert_time(int(record.start))
            end_ns = self._convert_time(int(record.end))
            return {
                "kind": "kernel",
                "device_id": int(record.device_id),
                "context_id": int(record.context_id),
                "stream_id": int(record.stream_id),
                "correlation_id": int(record.correlation_id),
                "graph_node_id": int(record.graph_node_id),
                "graph_id": int(record.graph_id),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "annotation": annotation,
                "name": kernel_name,
            }

        if kind == _CUPTI_ACTIVITY_KIND_MEMCPY:
            record = cc.ActivityMemcpy6.from_ptr(record_addr, readonly=True)
            annotation = self.annotation_resolver(
                int(record.graph_node_id),
                _RECORD_KIND_MEMCPY,
                int(record.correlation_id),
            )
            start_ns = self._convert_time(int(record.start))
            end_ns = self._convert_time(int(record.end))
            return {
                "kind": "gpu_memcpy",
                "device_id": int(record.device_id),
                "context_id": int(record.context_id),
                "stream_id": int(record.stream_id),
                "correlation_id": int(record.correlation_id),
                "runtime_correlation_id": int(record.runtime_correlation_id),
                "graph_node_id": int(record.graph_node_id),
                "graph_id": int(record.graph_id),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "bytes": int(record.bytes),
                "copy_kind": int(record.copy_kind),
                "src_kind": int(record.src_kind),
                "dst_kind": int(record.dst_kind),
                "flags": int(record.flags_),
                "annotation": annotation,
                "name": "Memcpy",
            }

        if kind == _CUPTI_ACTIVITY_KIND_MEMSET:
            record = cc.ActivityMemset4.from_ptr(record_addr, readonly=True)
            annotation = self.annotation_resolver(
                int(record.graph_node_id),
                _RECORD_KIND_MEMSET,
                int(record.correlation_id),
            )
            start_ns = self._convert_time(int(record.start))
            end_ns = self._convert_time(int(record.end))
            return {
                "kind": "gpu_memset",
                "device_id": int(record.device_id),
                "context_id": int(record.context_id),
                "stream_id": int(record.stream_id),
                "correlation_id": int(record.correlation_id),
                "graph_node_id": int(record.graph_node_id),
                "graph_id": int(record.graph_id),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "bytes": int(record.bytes),
                "value": int(record.value),
                "memory_kind": int(record.memory_kind),
                "flags": int(record.flags_),
                "annotation": annotation,
                "name": "Memset",
            }

        if kind in (_CUPTI_ACTIVITY_KIND_RUNTIME, _CUPTI_ACTIVITY_KIND_DRIVER):
            record = cc.ActivityAPI.from_ptr(record_addr, readonly=True)
            start_ns = self._convert_time(int(record.start))
            end_ns = self._convert_time(int(record.end))
            return {
                "kind": "cuda_runtime"
                if kind == _CUPTI_ACTIVITY_KIND_RUNTIME
                else "cuda_driver",
                "cbid": int(record.cbid),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "process_id": int(record.process_id),
                "thread_id": int(record.thread_id),
                "correlation_id": int(record.correlation_id),
                "return_value": int(record.return_value),
                "name": f"cbid_{int(record.cbid)}",
            }

        if kind == _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
            record = cc.ActivityExternalCorrelation.from_ptr(record_addr, readonly=True)
            return {
                "kind": "external_correlation",
                "external_kind": int(record.external_kind),
                "external_id": int(record.external_id),
                "correlation_id": int(record.correlation_id),
                "name": "external_correlation",
            }

        if kind == _CUPTI_ACTIVITY_KIND_OVERHEAD:
            record = cc.ActivityOverhead3.from_ptr(record_addr, readonly=True)
            start_ns = self._convert_time(int(record.start))
            end_ns = self._convert_time(int(record.end))
            overhead_kind = int(record.overhead_kind)
            return {
                "kind": "overhead",
                "overhead_kind": overhead_kind,
                "object_kind": int(record.object_kind),
                "object_id": 0,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "correlation_id": int(record.correlation_id),
                "name": _OVERHEAD_KIND_NAMES.get(
                    overhead_kind,
                    f"overhead_{overhead_kind}",
                ),
            }

        if kind == _CUPTI_ACTIVITY_KIND_CUDA_EVENT:
            record = cc.ActivityCudaEvent2.from_ptr(record_addr, readonly=True)
            event_ts = self._convert_time(int(record.device_timestamp))
            return {
                "kind": "cuda_event",
                "device_id": int(record.device_id),
                "context_id": int(record.context_id),
                "stream_id": int(record.stream_id),
                "event_id": int(record.event_id),
                "correlation_id": int(record.correlation_id),
                "device_timestamp_ns": event_ts,
                "cuda_event_sync_id": int(record.cuda_event_sync_id),
                "name": "cuda_event",
            }

        if kind == _CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
            record = cc.ActivitySynchronization2.from_ptr(record_addr, readonly=True)
            start_ns = self._convert_time(int(record.start))
            end_ns = self._convert_time(int(record.end))
            sync_type = int(record.type)
            return {
                "kind": "cuda_sync",
                "sync_type": sync_type,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "correlation_id": int(record.correlation_id),
                "context_id": int(record.context_id),
                "stream_id": int(record.stream_id),
                "event_id": int(record.cuda_event_id),
                "cuda_event_sync_id": int(record.cuda_event_sync_id),
                "return_value": int(record.return_value),
                "name": f"sync_{sync_type}",
            }

        return None

    def _write_raw_buffer(
        self, ctx: int, stream_id: int, buffer_ptr: int, valid_size: int
    ) -> None:
        if self._raw_buffers_fp is None:
            raise RuntimeError("raw buffer file is not open")
        approx_ns = int(getattr(_PY_PROFILER, "_get_approximate_time")())  # noqa: B009
        unix_ns = self._convert_time(approx_ns)
        header = _RAW_CHUNK_HEADER.pack(
            _RAW_CHUNK_MAGIC,
            _RAW_CHUNK_VERSION,
            self._raw_chunk_count,
            int(stream_id),
            int(valid_size),
            int(ctx),
            approx_ns,
            unix_ns,
        )
        self._raw_buffers_fp.write(header)
        self._raw_buffers_fp.write(ctypes.string_at(buffer_ptr, valid_size))
        self._raw_chunk_count += 1


_monitor_singleton: CuptiMonitor | None = None
_hes_enabled = False
_atexit_registered = False


def enable_hes_early() -> None:
    global _hes_enabled
    if _hes_enabled:
        return
    if torch.cuda.is_initialized() or _has_active_cuda_context():
        raise RuntimeError(
            "enable_hes_early() must be called before CUDA context creation"
        )
    from cuda.bindings import driver as cuda_driver  # pyrefly: ignore[missing-import]

    rc = cuda_driver.cuInit(0)[0]
    if rc != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cuInit failed with rc={rc}")

    # Do not use cupti-python's activity_enable_hw_trace() here. After torch is
    # imported, that path causes subsequent cuptiActivityRegisterCallbacks() to
    # fail with CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED in this process,
    # while the direct ctypes call below works.
    lib = ctypes.CDLL(_find_cupti_library())
    lib.cuptiActivityEnableHWTrace.argtypes = [ctypes.c_uint8]
    lib.cuptiActivityEnableHWTrace.restype = ctypes.c_int
    rc = lib.cuptiActivityEnableHWTrace(1)
    if rc != _CUPTI_SUCCESS:
        raise _CuptiError(f"cuptiActivityEnableHWTrace failed with rc={rc}")
    _hes_enabled = True


def is_hes_enabled() -> bool:
    return _hes_enabled


def get_monitor() -> CuptiMonitor | None:
    return _monitor_singleton


def start_collection(
    output_dir: str | os.PathLike[str],
    *,
    activities: Iterable[int] | None = None,
    buffer_size: int = _DEFAULT_BUFFER_SIZE,
    flush_period_s: float = _DEFAULT_FLUSH_PERIOD_S,
    annotation_resolver: Callable[[int, int, int], Any | None] | None = None,
) -> CuptiMonitor:
    global _monitor_singleton, _atexit_registered
    if _monitor_singleton is not None:
        raise RuntimeError("CUPTI monitor collection is already active")
    monitor = CuptiMonitor(
        output_dir,
        activities=activities,
        buffer_size=buffer_size,
        flush_period_s=flush_period_s,
        annotation_resolver=annotation_resolver,
    )
    monitor.start()
    _monitor_singleton = monitor
    if not _atexit_registered:
        atexit.register(_stop_collection_atexit)
        _atexit_registered = True
    return monitor


def stop_collection() -> dict[str, Any] | None:
    global _monitor_singleton
    monitor = _monitor_singleton
    if monitor is None:
        return None
    try:
        monitor.stop()
        return monitor.stats()
    finally:
        _monitor_singleton = None


def monitor_stats() -> dict[str, Any] | None:
    if _monitor_singleton is None:
        return None
    return _monitor_singleton.stats()


def begin_trace_window(
    activities: Iterable[int] | None = None,
) -> tuple[int, ...]:
    if _monitor_singleton is None:
        raise RuntimeError("CUPTI monitor collection is not active")
    return _monitor_singleton.begin_trace_window(activities)


def prepare_trace_window(
    activities: Iterable[int] | None = None,
) -> tuple[int, ...]:
    if _monitor_singleton is None:
        raise RuntimeError("CUPTI monitor collection is not active")
    return _monitor_singleton.prepare_trace_window(activities)


def start_trace_window() -> None:
    if _monitor_singleton is None:
        raise RuntimeError("CUPTI monitor collection is not active")
    _monitor_singleton.start_trace_window()


def end_trace_window() -> dict[str, Any]:
    if _monitor_singleton is None:
        raise RuntimeError("CUPTI monitor collection is not active")
    return _monitor_singleton.end_trace_window()


def push_user_annotation(name: str) -> int | None:
    if _monitor_singleton is None:
        return None
    return _monitor_singleton.push_user_annotation(name)


def pop_user_annotation() -> int | None:
    if _monitor_singleton is None:
        return None
    return _monitor_singleton.pop_user_annotation()


def _stop_collection_atexit() -> None:
    try:
        stop_collection()
    except Exception:
        pass
