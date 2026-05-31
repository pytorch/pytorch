# mypy: allow-untyped-defs
from __future__ import annotations

import atexit
import ctypes
import ctypes.util
import glob
import json
import os
import queue
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch


_LIBCUPTI_PATH_ENV = "TORCH_CUPTI_MONITOR_LIBCUPTI_PATH"

_CUPTI_SUCCESS = 0
_CUPTI_ERROR_MAX_LIMIT_REACHED = 12

_CUPTI_ACTIVITY_KIND_MEMCPY = 1
_CUPTI_ACTIVITY_KIND_MEMSET = 2
_CUPTI_ACTIVITY_KIND_DRIVER = 4
_CUPTI_ACTIVITY_KIND_RUNTIME = 5
_CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10
_CUPTI_ACTIVITY_KIND_OVERHEAD = 17
_CUPTI_ACTIVITY_KIND_CUDA_EVENT = 36
_CUPTI_ACTIVITY_KIND_SYNCHRONIZATION = 38
_CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION = 39

_CUPTI_ACTIVITY_FLAG_FLUSH_FORCED = 1

_DEFAULT_BUFFER_SIZE = 4 * 1024 * 1024
_DEFAULT_FLUSH_PERIOD_S = 1.0

_RECORD_FILE = "cupti_monitor_records.bin"
_ANNOTATION_FILE = "cupti_monitor_annotations.bin"
_META_FILE = "cupti_monitor_meta.bin"

_CHUNK_MAGIC = b"CUPMREC1"
_CHUNK_VERSION = 4
_CHUNK_HEADER = struct.Struct("<8sIIQQQ")
_RECORD_STRUCT = struct.Struct("<IIIIIQQQQQIIII")
_ANNOTATION_HEADER = struct.Struct("<QI")
_META_MAGIC = b"CUPMMETA"
_META_HEADER = struct.Struct("<8sIIII")

_RECORD_KIND_KERNEL = 1
_RECORD_KIND_MEMCPY = 2
_RECORD_KIND_MEMSET = 3

_DEFAULT_ALWAYS_ON_ACTIVITIES = (
    _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    _CUPTI_ACTIVITY_KIND_MEMCPY,
    _CUPTI_ACTIVITY_KIND_MEMSET,
)

_DEFAULT_TRACE_WINDOW_ACTIVITIES = (
    _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    _CUPTI_ACTIVITY_KIND_MEMCPY,
    _CUPTI_ACTIVITY_KIND_MEMSET,
    _CUPTI_ACTIVITY_KIND_RUNTIME,
    _CUPTI_ACTIVITY_KIND_DRIVER,
    _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION,
    _CUPTI_ACTIVITY_KIND_OVERHEAD,
    _CUPTI_ACTIVITY_KIND_CUDA_EVENT,
    _CUPTI_ACTIVITY_KIND_SYNCHRONIZATION,
)

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


def _find_cupti_library() -> str:
    override = os.environ.get(_LIBCUPTI_PATH_ENV)
    if override:
        return override

    found = ctypes.util.find_library("cupti")
    if found:
        return found

    candidate_patterns = [
        os.path.join(path, "nvidia", "cuda_cupti", "lib", "libcupti.so.*[0-9]")
        for path in sys.path
    ]
    candidate_patterns.extend(
        os.path.join(path, "nvidia", "cu*", "lib", "libcupti.so.*[0-9]")
        for path in sys.path
    )
    candidate_patterns.extend(
        os.path.join(path, "cuda_cupti", "lib", "libcupti.so.*[0-9]")
        for path in sys.path
    )

    for pattern in candidate_patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return "libcupti.so"

def _has_active_cuda_context() -> bool:
    try:
        from cuda.bindings import driver as cuda_driver
    except ImportError:
        return False
    rc, ctx = cuda_driver.cuCtxGetCurrent()
    if rc == cuda_driver.CUresult.CUDA_SUCCESS:
        return ctx is not None
    if rc == cuda_driver.CUresult.CUDA_ERROR_NOT_INITIALIZED:
        return False
    raise RuntimeError(f"cuCtxGetCurrent failed with rc={rc}")
class _CuptiError(RuntimeError):
    pass


class _CuptiActivity(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int)]


class _CuptiActivityKernel(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("cacheConfig", ctypes.c_uint8),
        ("sharedMemoryConfig", ctypes.c_uint8),
        ("registersPerThread", ctypes.c_uint16),
        ("partitionedGlobalCacheRequested", ctypes.c_int),
        ("partitionedGlobalCacheExecuted", ctypes.c_int),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("completed", ctypes.c_uint64),
        ("deviceId", ctypes.c_uint32),
        ("contextId", ctypes.c_uint32),
        ("streamId", ctypes.c_uint32),
        ("gridX", ctypes.c_int32),
        ("gridY", ctypes.c_int32),
        ("gridZ", ctypes.c_int32),
        ("blockX", ctypes.c_int32),
        ("blockY", ctypes.c_int32),
        ("blockZ", ctypes.c_int32),
        ("staticSharedMemory", ctypes.c_int32),
        ("dynamicSharedMemory", ctypes.c_int32),
        ("localMemoryPerThread", ctypes.c_uint32),
        ("localMemoryTotal", ctypes.c_uint32),
        ("correlationId", ctypes.c_uint32),
        ("gridId", ctypes.c_int64),
        ("name", ctypes.c_void_p),
        ("reserved0", ctypes.c_void_p),
        ("queued", ctypes.c_uint64),
        ("submitted", ctypes.c_uint64),
        ("launchType", ctypes.c_uint8),
        ("isSharedMemoryCarveoutRequested", ctypes.c_uint8),
        ("sharedMemoryCarveoutRequested", ctypes.c_uint8),
        ("padding", ctypes.c_uint8),
        ("sharedMemoryExecuted", ctypes.c_uint32),
        ("graphNodeId", ctypes.c_uint64),
        ("shmemLimitConfig", ctypes.c_int),
        ("graphId", ctypes.c_uint32),
    ]


class _CuptiActivityMemcpy(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("copyKind", ctypes.c_uint8),
        ("srcKind", ctypes.c_uint8),
        ("dstKind", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("bytes", ctypes.c_uint64),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("deviceId", ctypes.c_uint32),
        ("contextId", ctypes.c_uint32),
        ("streamId", ctypes.c_uint32),
        ("correlationId", ctypes.c_uint32),
        ("runtimeCorrelationId", ctypes.c_uint32),
        ("reserved0", ctypes.c_void_p),
        ("graphNodeId", ctypes.c_uint64),
        ("graphId", ctypes.c_uint32),
    ]


class _CuptiActivityMemset(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("value", ctypes.c_uint32),
        ("bytes", ctypes.c_uint64),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("deviceId", ctypes.c_uint32),
        ("contextId", ctypes.c_uint32),
        ("streamId", ctypes.c_uint32),
        ("correlationId", ctypes.c_uint32),
        ("flags", ctypes.c_uint16),
        ("memoryKind", ctypes.c_uint16),
        ("reserved0", ctypes.c_void_p),
        ("graphNodeId", ctypes.c_uint64),
        ("graphId", ctypes.c_uint32),
    ]


class _CuptiActivityAPI(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("cbid", ctypes.c_uint32),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("processId", ctypes.c_uint32),
        ("threadId", ctypes.c_uint32),
        ("correlationId", ctypes.c_uint32),
        ("returnValue", ctypes.c_uint32),
    ]


class _CuptiActivityExternalCorrelation(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("externalKind", ctypes.c_int),
        ("externalId", ctypes.c_uint64),
        ("correlationId", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _CuptiActivityOverhead(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("overheadKind", ctypes.c_int),
        ("objectKind", ctypes.c_int),
        ("objectId", ctypes.c_uint64),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("correlationId", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("overheadData", ctypes.c_void_p),
    ]


class _CuptiActivityCudaEvent(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("correlationId", ctypes.c_uint32),
        ("contextId", ctypes.c_uint32),
        ("streamId", ctypes.c_uint32),
        ("eventId", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
        ("deviceId", ctypes.c_uint32),
        ("pad2", ctypes.c_uint32),
        ("reserved0", ctypes.c_void_p),
        ("deviceTimestamp", ctypes.c_uint64),
        ("cudaEventSyncId", ctypes.c_uint64),
    ]


class _CuptiActivitySynchronization(ctypes.Structure):
    _fields_ = [
        ("kind", ctypes.c_int),
        ("type", ctypes.c_int),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("correlationId", ctypes.c_uint32),
        ("contextId", ctypes.c_uint32),
        ("streamId", ctypes.c_uint32),
        ("cudaEventId", ctypes.c_uint32),
        ("cudaEventSyncId", ctypes.c_uint64),
        ("returnValue", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]


def _normalize_annotation(annotation: Any) -> bytes:
    if annotation is None:
        return b""
    try:
        return json.dumps(annotation, sort_keys=True, separators=(",", ":")).encode()
    except TypeError:
        return str(annotation).encode()


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
        self.output_dir = Path(output_dir)
        self.activities = tuple(activities or _DEFAULT_ALWAYS_ON_ACTIVITIES)
        self.buffer_size = buffer_size
        self.flush_period_s = flush_period_s
        self.annotation_resolver = (
            annotation_resolver or _default_graph_annotation_resolver
        )

        self._lib = ctypes.CDLL(_find_cupti_library())
        self._setup_prototypes()

        self._lock = threading.Lock()
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
        self._trace_window_events: list[dict[str, Any]] = []
        self._trace_window_extra_activities: tuple[int, ...] = ()

        self._records_fp = None
        self._annotations_fp = None

        self._buffers_requested = 0
        self._buffers_completed = 0
        self._valid_bytes = 0
        self._max_outstanding = 0
        self._dropped_records = 0
        self._chunk_count = 0
        self._annotation_ids: dict[bytes, int] = {}
        self._next_annotation_id = 1

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
        self._lib.cuptiActivityRegisterCallbacks.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._lib.cuptiActivityRegisterCallbacks.restype = ctypes.c_int
        self._lib.cuptiActivityEnable.argtypes = [ctypes.c_int]
        self._lib.cuptiActivityEnable.restype = ctypes.c_int
        self._lib.cuptiActivityDisable.argtypes = [ctypes.c_int]
        self._lib.cuptiActivityDisable.restype = ctypes.c_int
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
        self._lib.cuptiGetResultString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._lib.cuptiGetResultString.restype = ctypes.c_int

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

    def start(self) -> None:
        if self._started:
            raise RuntimeError("CUPTI monitor is already started")
        self.register_callbacks()
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
        for activity in self.activities:
            self._check(self._lib.cuptiActivityEnable(activity), "cuptiActivityEnable")
            self._enabled_activities.add(activity)
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

    def flush(self, *, forced: bool = False) -> None:
        flag = _CUPTI_ACTIVITY_FLAG_FLUSH_FORCED if forced else 0
        self._check(self._lib.cuptiActivityFlushAll(flag), "cuptiActivityFlushAll")

    def enable_activities(self, activities: Iterable[int]) -> tuple[int, ...]:
        newly_enabled = []
        for activity in activities:
            if activity in self._enabled_activities:
                continue
            self._check(self._lib.cuptiActivityEnable(activity), "cuptiActivityEnable")
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

    def begin_trace_window(
        self, activities: Iterable[int] | None = None
    ) -> tuple[int, ...]:
        if not self._started:
            raise RuntimeError("CUPTI monitor must be started before begin_trace_window")
        if self._trace_window_active:
            raise RuntimeError("A trace window is already active")
        activities = tuple(activities or _DEFAULT_TRACE_WINDOW_ACTIVITIES)
        newly_enabled = self.enable_activities(activities)
        with self._lock:
            self._trace_window_events = []
            self._trace_window_active = True
            self._trace_window_extra_activities = newly_enabled
        return newly_enabled

    def end_trace_window(self) -> dict[str, Any]:
        if not self._trace_window_active:
            raise RuntimeError("No active trace window")
        self.flush(forced=True)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._completed_queue.empty():
                break
            time.sleep(0.01)
        with self._lock:
            events = list(self._trace_window_events)
            extra = self._trace_window_extra_activities
            self._trace_window_events = []
            self._trace_window_active = False
            self._trace_window_extra_activities = ()
        self.disable_activities(extra)
        return {"events": events, "extra_activities": list(extra)}

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
                "chunks_written": self._chunk_count,
                "output_dir": str(self.output_dir),
                "trace_window_active": self._trace_window_active,
            }

    def _open_outputs(self) -> None:
        self._records_fp = open(self.output_dir / _RECORD_FILE, "ab", buffering=0)
        self._annotations_fp = open(
            self.output_dir / _ANNOTATION_FILE, "ab", buffering=0
        )

    def _close_outputs(self) -> None:
        if self._records_fp is not None:
            self._records_fp.close()
            self._records_fp = None
        if self._annotations_fp is not None:
            self._annotations_fp.close()
            self._annotations_fp = None

    def _write_meta_file(self) -> None:
        lib_path = _find_cupti_library().encode()
        activities = list(self.activities)
        with open(self.output_dir / _META_FILE, "wb") as fp:
            fp.write(
                _META_HEADER.pack(
                    _META_MAGIC,
                    _CHUNK_VERSION,
                    len(activities),
                    self.buffer_size,
                    int(self.flush_period_s * 1e9),
                )
            )
            for activity in activities:
                fp.write(struct.pack("<I", activity))
            fp.write(struct.pack("<Q", time.time_ns()))
            fp.write(struct.pack("<I", len(lib_path)))
            fp.write(lib_path)

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
                self._process_completed_buffer(ctx, stream_id, buffer_ptr, valid_size)
        except BaseException as exc:
            self._worker_error = exc
            self._worker_stop.set()

    def _process_completed_buffer(
        self, ctx: int, stream_id: int, buffer_ptr: int, valid_size: int
    ) -> None:
        record_ptr = ctypes.c_void_p()
        chunk_records: list[bytes] = []
        min_ts = 0
        max_ts = 0
        while True:
            rc = self._lib.cuptiActivityGetNextRecord(
                ctypes.c_void_p(buffer_ptr),
                ctypes.c_size_t(valid_size),
                ctypes.byref(record_ptr),
            )
            if rc == _CUPTI_SUCCESS:
                record_bytes, start_ns, end_ns, trace_event = self._decode_record(
                    record_ptr.value
                )
                if record_bytes is not None:
                    chunk_records.append(record_bytes)
                    if start_ns:
                        min_ts = start_ns if min_ts == 0 else min(min_ts, start_ns)
                    max_ts = max(max_ts, end_ns)
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

        if chunk_records:
            self._write_chunk(chunk_records, min_ts, max_ts)

        with self._lock:
            self._free_buffers.append(buffer_ptr)

    def _annotation_id_for_record(
        self, graph_node_id: int, record_kind: int, correlation_id: int
    ) -> int:
        annotation = self.annotation_resolver(graph_node_id, record_kind, correlation_id)
        if annotation is None:
            return 0
        key = _normalize_annotation(annotation)
        if not key:
            return 0
        annotation_id = self._annotation_ids.get(key)
        if annotation_id is not None:
            return annotation_id
        annotation_id = self._next_annotation_id
        self._next_annotation_id += 1
        self._annotation_ids[key] = annotation_id
        if self._annotations_fp is None:
            raise RuntimeError("annotations file is not open")
        self._annotations_fp.write(_ANNOTATION_HEADER.pack(annotation_id, len(key)))
        self._annotations_fp.write(key)
        return annotation_id

    def _decode_record(
        self, record_addr: int
    ) -> tuple[bytes | None, int, int, dict[str, Any] | None]:
        activity = ctypes.cast(record_addr, ctypes.POINTER(_CuptiActivity)).contents
        kind = int(activity.kind)
        if kind == _CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
            record = ctypes.cast(
                record_addr, ctypes.POINTER(_CuptiActivityKernel)
            ).contents
            annotation = self.annotation_resolver(
                int(record.graphNodeId),
                _RECORD_KIND_KERNEL,
                int(record.correlationId),
            )
            annotation_id = self._annotation_id_for_record(
                int(record.graphNodeId),
                _RECORD_KIND_KERNEL,
                int(record.correlationId),
            )
            kernel_name = _decode_c_string(
                int(record.name) if record.name else 0,
                "kernel",
            )
            kernel_name = _demangle_symbol(kernel_name)
            payload = _RECORD_STRUCT.pack(
                _RECORD_KIND_KERNEL,
                int(record.deviceId),
                int(record.contextId),
                int(record.streamId),
                int(record.correlationId),
                annotation_id,
                int(record.graphNodeId),
                int(record.start),
                int(record.end),
                0,
                0,
                0,
                int(record.graphId),
                0,
            )
            trace_event = {
                "kind": "kernel",
                "device_id": int(record.deviceId),
                "context_id": int(record.contextId),
                "stream_id": int(record.streamId),
                "correlation_id": int(record.correlationId),
                "graph_node_id": int(record.graphNodeId),
                "graph_id": int(record.graphId),
                "start_ns": int(record.start),
                "end_ns": int(record.end),
                "annotation_id": annotation_id,
                "annotation": annotation,
                "name": kernel_name,
            }
            return payload, int(record.start), int(record.end), trace_event

        if kind == _CUPTI_ACTIVITY_KIND_MEMCPY:
            record = ctypes.cast(
                record_addr, ctypes.POINTER(_CuptiActivityMemcpy)
            ).contents
            annotation = self.annotation_resolver(
                int(record.graphNodeId),
                _RECORD_KIND_MEMCPY,
                int(record.correlationId),
            )
            annotation_id = self._annotation_id_for_record(
                int(record.graphNodeId),
                _RECORD_KIND_MEMCPY,
                int(record.correlationId),
            )
            aux = (
                int(record.copyKind)
                | (int(record.srcKind) << 8)
                | (int(record.dstKind) << 16)
                | (int(record.flags) << 24)
            )
            payload = _RECORD_STRUCT.pack(
                _RECORD_KIND_MEMCPY,
                int(record.deviceId),
                int(record.contextId),
                int(record.streamId),
                int(record.correlationId),
                annotation_id,
                int(record.graphNodeId),
                int(record.start),
                int(record.end),
                int(record.bytes),
                int(record.runtimeCorrelationId),
                aux,
                0,
                int(record.graphId),
            )
            trace_event = {
                "kind": "gpu_memcpy",
                "device_id": int(record.deviceId),
                "context_id": int(record.contextId),
                "stream_id": int(record.streamId),
                "correlation_id": int(record.correlationId),
                "runtime_correlation_id": int(record.runtimeCorrelationId),
                "graph_node_id": int(record.graphNodeId),
                "graph_id": int(record.graphId),
                "start_ns": int(record.start),
                "end_ns": int(record.end),
                "bytes": int(record.bytes),
                "copy_kind": int(record.copyKind),
                "src_kind": int(record.srcKind),
                "dst_kind": int(record.dstKind),
                "flags": int(record.flags),
                "annotation_id": annotation_id,
                "annotation": annotation,
                "name": "Memcpy",
            }
            return payload, int(record.start), int(record.end), trace_event

        if kind == _CUPTI_ACTIVITY_KIND_MEMSET:
            record = ctypes.cast(
                record_addr, ctypes.POINTER(_CuptiActivityMemset)
            ).contents
            annotation = self.annotation_resolver(
                int(record.graphNodeId),
                _RECORD_KIND_MEMSET,
                int(record.correlationId),
            )
            annotation_id = self._annotation_id_for_record(
                int(record.graphNodeId),
                _RECORD_KIND_MEMSET,
                int(record.correlationId),
            )
            aux = int(record.value)
            aux2 = int(record.memoryKind) | (int(record.flags) << 16)
            payload = _RECORD_STRUCT.pack(
                _RECORD_KIND_MEMSET,
                int(record.deviceId),
                int(record.contextId),
                int(record.streamId),
                int(record.correlationId),
                annotation_id,
                int(record.graphNodeId),
                int(record.start),
                int(record.end),
                int(record.bytes),
                0,
                aux,
                aux2,
                int(record.graphId),
            )
            trace_event = {
                "kind": "gpu_memset",
                "device_id": int(record.deviceId),
                "context_id": int(record.contextId),
                "stream_id": int(record.streamId),
                "correlation_id": int(record.correlationId),
                "graph_node_id": int(record.graphNodeId),
                "graph_id": int(record.graphId),
                "start_ns": int(record.start),
                "end_ns": int(record.end),
                "bytes": int(record.bytes),
                "value": int(record.value),
                "memory_kind": int(record.memoryKind),
                "flags": int(record.flags),
                "annotation_id": annotation_id,
                "annotation": annotation,
                "name": "Memset",
            }
            return payload, int(record.start), int(record.end), trace_event

        if kind in (_CUPTI_ACTIVITY_KIND_RUNTIME, _CUPTI_ACTIVITY_KIND_DRIVER):
            record = ctypes.cast(record_addr, ctypes.POINTER(_CuptiActivityAPI)).contents
            trace_event = {
                "kind": "cuda_runtime" if kind == _CUPTI_ACTIVITY_KIND_RUNTIME else "cuda_driver",
                "cbid": int(record.cbid),
                "start_ns": int(record.start),
                "end_ns": int(record.end),
                "process_id": int(record.processId),
                "thread_id": int(record.threadId),
                "correlation_id": int(record.correlationId),
                "return_value": int(record.returnValue),
                "name": f"cbid_{int(record.cbid)}",
            }
            return None, int(record.start), int(record.end), trace_event

        if kind == _CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
            record = ctypes.cast(
                record_addr, ctypes.POINTER(_CuptiActivityExternalCorrelation)
            ).contents
            trace_event = {
                "kind": "external_correlation",
                "external_kind": int(record.externalKind),
                "external_id": int(record.externalId),
                "correlation_id": int(record.correlationId),
                "name": "external_correlation",
            }
            return None, 0, 0, trace_event

        if kind == _CUPTI_ACTIVITY_KIND_OVERHEAD:
            record = ctypes.cast(
                record_addr, ctypes.POINTER(_CuptiActivityOverhead)
            ).contents
            trace_event = {
                "kind": "overhead",
                "overhead_kind": int(record.overheadKind),
                "object_kind": int(record.objectKind),
                "object_id": int(record.objectId),
                "start_ns": int(record.start),
                "end_ns": int(record.end),
                "correlation_id": int(record.correlationId),
                "name": _OVERHEAD_KIND_NAMES.get(
                    int(record.overheadKind),
                    f"overhead_{int(record.overheadKind)}",
                ),
            }
            return None, int(record.start), int(record.end), trace_event

        if kind == _CUPTI_ACTIVITY_KIND_CUDA_EVENT:
            record = ctypes.cast(
                record_addr, ctypes.POINTER(_CuptiActivityCudaEvent)
            ).contents
            trace_event = {
                "kind": "cuda_event",
                "device_id": int(record.deviceId),
                "context_id": int(record.contextId),
                "stream_id": int(record.streamId),
                "event_id": int(record.eventId),
                "correlation_id": int(record.correlationId),
                "device_timestamp_ns": int(record.deviceTimestamp),
                "cuda_event_sync_id": int(record.cudaEventSyncId),
                "name": "cuda_event",
            }
            return None, int(record.deviceTimestamp), int(record.deviceTimestamp), trace_event

        if kind == _CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
            record = ctypes.cast(
                record_addr, ctypes.POINTER(_CuptiActivitySynchronization)
            ).contents
            trace_event = {
                "kind": "cuda_sync",
                "sync_type": int(record.type),
                "start_ns": int(record.start),
                "end_ns": int(record.end),
                "correlation_id": int(record.correlationId),
                "context_id": int(record.contextId),
                "stream_id": int(record.streamId),
                "event_id": int(record.cudaEventId),
                "cuda_event_sync_id": int(record.cudaEventSyncId),
                "return_value": int(record.returnValue),
                "name": f"sync_{int(record.type)}",
            }
            return None, int(record.start), int(record.end), trace_event

        return None, 0, 0, None

    def _write_chunk(self, records: list[bytes], min_ts: int, max_ts: int) -> None:
        if self._records_fp is None:
            raise RuntimeError("records file is not open")
        payload = b"".join(records)
        header = _CHUNK_HEADER.pack(
            _CHUNK_MAGIC,
            _CHUNK_VERSION,
            len(records),
            len(payload),
            min_ts,
            max_ts,
        )
        self._records_fp.write(header)
        self._records_fp.write(payload)
        self._chunk_count += 1


_monitor_singleton: CuptiMonitor | None = None
_hes_enabled = False
_atexit_registered = False


def enable_hes_early() -> None:
    global _hes_enabled
    if _hes_enabled:
        return
    if torch.cuda.is_initialized() or _has_active_cuda_context():
        raise RuntimeError("enable_hes_early() must be called before CUDA context creation")
    lib = ctypes.CDLL(_find_cupti_library())
    lib.cuptiActivityEnableHWTrace.argtypes = [ctypes.c_uint8]
    lib.cuptiActivityEnableHWTrace.restype = ctypes.c_int
    rc = lib.cuptiActivityEnableHWTrace(1)
    if rc != _CUPTI_SUCCESS:
        try:
            libcuda = ctypes.CDLL("libcuda.so.1")
            libcuda.cuInit.argtypes = [ctypes.c_uint32]
            libcuda.cuInit.restype = ctypes.c_int
            if libcuda.cuInit(0) == 0:
                rc = lib.cuptiActivityEnableHWTrace(1)
        except OSError:
            pass
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


def end_trace_window() -> dict[str, Any]:
    if _monitor_singleton is None:
        raise RuntimeError("CUPTI monitor collection is not active")
    return _monitor_singleton.end_trace_window()


def _stop_collection_atexit() -> None:
    try:
        stop_collection()
    except Exception:
        pass
