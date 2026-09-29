"""Order host-driven transport operations on CUDA streams.

A host callback enqueued on the stream hands the transfer to a progress thread,
which submits it and signals a pinned completion flag. A stream memory wait
holds later stream work until that flag is set, so no kernel occupies SMs while
the transfer runs. The callback itself never blocks: CUDA forbids CUDA calls
from host callbacks, and blocking on native CUDA work can deadlock graph replay.
"""

from __future__ import annotations

import ctypes
import logging
import os
import queue
import sys
import threading
from contextlib import contextmanager
from typing import Any, cast, TYPE_CHECKING

import torch
from torch.distributed import Work


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


logger = logging.getLogger(__name__)

_NUM_SLOTS = 64
_CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118
_CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100
_CU_STREAM_WAIT_VALUE_GEQ = 0

_HostFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_libcuda: ctypes.CDLL | None = None


def _driver() -> ctypes.CDLL:
    global _libcuda
    if _libcuda is None:
        if sys.platform != "linux":
            raise RuntimeError("CUDA stream transfers require Linux")
        # cuLaunchHostFunc is called directly: callback userdata owned by
        # binding libraries may be released after a captured node first runs.
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, _HostFn, ctypes.c_void_p]
        lib.cuStreamWaitValue32_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint32,
            ctypes.c_uint,
        ]
        lib.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        _libcuda = lib
    return _libcuda


def _check(result: int, what: str) -> None:
    if result:
        raise RuntimeError(f"{what} failed with CUDA driver error {result}")


def _fatal(message: str, error: BaseException) -> None:
    # Consumers already wait on this transfer. Exiting is the only way to keep
    # them from observing incomplete data.
    logger.critical(message, exc_info=error)
    os._exit(1)


class _CudaStreamOrdering:
    """Orders transfers on one CUDA stream; owns callbacks and captured graphs."""

    def __init__(self, stream: torch.cuda.Stream) -> None:
        driver = _driver()
        ordering = ctypes.c_int()
        _check(
            driver.cuDeviceGetAttribute(
                ctypes.byref(ordering),
                _CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                stream.device.index,
            ),
            "cuDeviceGetAttribute",
        )
        # Host-observed completion must imply that device consumers can read
        # remote writes without an explicit flush.
        if ordering.value < _CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER:
            raise RuntimeError(
                "CUDA stream transfers require native GPUDirect RDMA write ordering"
            )
        self._driver = driver
        self._stream = stream
        # Pinned memory is device-mapped at the same address under UVA.
        self._flags = torch.zeros(_NUM_SLOTS, dtype=torch.int32, pin_memory=True)
        self._free_slots = list(range(_NUM_SLOTS))
        self._lock = threading.Lock()
        self._closed = False
        self._capturing = False
        # Callback trampolines and views must outlive every launch that uses
        # them, not merely the completion of their transfer.
        # Entries are (retired event, view, slot, keepalive); captured entries
        # have no event and are released only on close.
        self._pending: list[tuple[torch.cuda.Event, Any, int, Any]] = []
        self._captured: list[tuple[None, Any, int, Any]] = []
        self._graphs: list[torch.cuda.CUDAGraph] = []
        self._queue: queue.SimpleQueue[
            tuple[Callable[[], None], ctypes.c_uint32] | None
        ] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._progress, daemon=True)
        self._thread.start()

    def _progress(self) -> None:
        # Unlike the CUDA callback thread, this thread may call CUDA.
        torch.cuda.set_device(self._stream.device)
        while (item := self._queue.get()) is not None:
            transfer, flag = item
            transfer()
            flag.value = 1

    def _flag(self, slot: int) -> ctypes.c_uint32:
        return ctypes.c_uint32.from_address(self._flags.data_ptr() + slot * 4)

    def _reap(self) -> None:
        pending = []
        for item in self._pending:
            if item[0].query():
                self._free_slots.append(item[2])
            else:
                pending.append(item)
        self._pending = pending

    def retained_views(self) -> list[Any]:
        with self._lock:
            return [item[1] for item in self._captured] + [
                item[1] for item in self._pending if not item[0].query()
            ]

    @contextmanager
    def capture(self) -> Iterator[torch.cuda.CUDAGraph]:
        if self._closed or self._capturing:
            raise RuntimeError("stream is closed or already capturing")
        graph = torch.cuda.CUDAGraph()
        self._graphs.append(graph)
        self._capturing = True
        try:
            with torch.cuda.graph(graph, stream=self._stream):
                yield graph
        finally:
            self._capturing = False

    def enqueue(self, submit: Callable[[], int | Work], view: Any) -> None:
        with self._lock, torch.cuda.stream(self._stream):
            if self._closed:
                raise RuntimeError("transport is closed")
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and not self._capturing:
                raise RuntimeError("capture transfers with Transport.cuda_graph()")
            if not capturing:
                self._reap()
            if not self._free_slots:
                raise RuntimeError(
                    "too many pending CUDA stream transfers; synchronize the stream"
                )
            slot = self._free_slots.pop()
            flag = self._flag(slot)
            flag.value = 0

            def transfer() -> None:
                try:
                    cast(Work, submit()).wait()
                except BaseException as error:
                    _fatal("CUDA stream transfer failed", error)

            @_HostFn
            def callback(_: int) -> None:
                try:
                    # Each graph replay resets its flag before the stream waits.
                    flag.value = 0
                    self._queue.put((transfer, flag))
                except BaseException as error:
                    _fatal("CUDA stream transfer callback failed", error)

            try:
                _check(
                    self._driver.cuLaunchHostFunc(
                        self._stream.cuda_stream, callback, None
                    ),
                    "cuLaunchHostFunc",
                )
            except BaseException:
                self._free_slots.append(slot)
                raise
            try:
                _check(
                    self._driver.cuStreamWaitValue32_v2(
                        self._stream.cuda_stream,
                        self._flags.data_ptr() + slot * 4,
                        1,
                        _CU_STREAM_WAIT_VALUE_GEQ,
                    ),
                    "cuStreamWaitValue32",
                )
            except BaseException as error:
                _fatal("CUDA stream transfer completion wait failed", error)
            if capturing:
                self._captured.append((None, view, slot, callback))
            else:
                retired = torch.cuda.Event()
                retired.record(self._stream)
                self._pending.append((retired, view, slot, callback))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._capturing:
                raise RuntimeError("cannot close during CUDA graph capture")
            # Graphs may replay on other streams; keep callbacks alive until
            # all launches finish and graph executables are destroyed.
            torch.cuda.synchronize(self._stream.device)
            for graph in self._graphs:
                graph.reset()
            self._graphs.clear()
            self._captured.clear()
            self._pending.clear()
            self._closed = True
            self._queue.put(None)
            self._thread.join()
