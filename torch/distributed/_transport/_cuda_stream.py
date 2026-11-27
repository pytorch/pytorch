"""Order host-driven transport operations on CUDA streams.

A host callback enqueued on the stream schedules the transfer on a shared
asyncio loop, which awaits it and signals a pinned completion flag. A stream memory wait
holds later stream work until that flag is set, so no kernel occupies SMs while
the transfer runs. The callback itself never blocks: CUDA forbids CUDA calls
from host callbacks, and blocking on native CUDA work can deadlock graph replay.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import sys
import threading
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


logger = logging.getLogger(__name__)

_NUM_SLOTS = 64
_CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118
_CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100
_CU_STREAM_WAIT_VALUE_GEQ = 0

_HostFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_libcuda: ctypes.CDLL | None = None
_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()
# Captured callbacks outlive their transport: a graph may replay after close.
_closed_captures: list[Any] = []


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


def _event_loop() -> asyncio.AbstractEventLoop:
    global _loop
    with _loop_lock:
        if _loop is None:
            # Unlike the CUDA callback thread, this loop may call CUDA.
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True).start()
        return _loop


def _check(result: int, what: str) -> None:
    if result:
        raise RuntimeError(f"{what} failed with CUDA driver error {result}")


def _fatal(message: str, error: BaseException) -> None:
    # Consumers already wait on this transfer. Exiting is the only way to keep
    # them from observing incomplete data.
    logger.critical(message, exc_info=error)
    os._exit(1)


class _CudaStreamOrdering:
    """Orders transfers on one CUDA stream and owns their callbacks."""

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
        self._loop = _event_loop()
        self._stream = stream
        # Pinned memory is device-mapped at the same address under UVA.
        self._flags = torch.zeros(_NUM_SLOTS, dtype=torch.int32, pin_memory=True)
        self._free_slots = list(range(_NUM_SLOTS))
        self._lock = threading.Lock()
        self._closed = False
        # Callback trampolines and views must outlive every launch that uses
        # them, not merely the completion of their transfer.
        # Entries are (retired event, view, slot, keepalive); captured entries
        # have no event and are released only on close.
        self._retained: list[tuple[torch.cuda.Event | None, Any, int, Any]] = []

    def _flag(self, slot: int) -> ctypes.c_uint32:
        return ctypes.c_uint32.from_address(self._flags.data_ptr() + slot * 4)

    def _reap(self) -> None:
        retained = []
        for item in self._retained:
            if item[0] is not None and item[0].query():
                self._free_slots.append(item[2])
            else:
                retained.append(item)
        self._retained = retained

    def retained_views(self) -> list[Any]:
        with self._lock:
            return [
                item[1]
                for item in self._retained
                if item[0] is None or not item[0].query()
            ]

    def enqueue(
        self, submit: Callable[[], Coroutine[Any, Any, None]], view: Any
    ) -> None:
        with self._lock, torch.cuda.stream(self._stream):
            if self._closed:
                raise RuntimeError("transport is closed")
            capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                self._reap()
            if not self._free_slots:
                raise RuntimeError(
                    "too many pending CUDA stream transfers; synchronize the stream"
                )
            slot = self._free_slots.pop()
            flag = self._flag(slot)
            flag.value = 0

            device = self._stream.device

            async def transfer() -> None:
                try:
                    torch.cuda.set_device(device)
                    await submit()
                except BaseException as error:
                    _fatal("CUDA stream transfer failed", error)
                flag.value = 1

            def start() -> None:
                self._loop.create_task(transfer())

            @_HostFn
            def callback(_: int) -> None:
                try:
                    if self._closed:
                        raise RuntimeError("CUDA graph replayed after transport close")
                    # Each graph replay resets its flag before the stream waits.
                    flag.value = 0
                    self._loop.call_soon_threadsafe(start)
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
            retired = None
            if not capturing:
                retired = torch.cuda.Event()
                retired.record(self._stream)
            self._retained.append((retired, view, slot, callback))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            torch.cuda.synchronize(self._stream.device)
            _closed_captures.extend(
                item[3] for item in self._retained if item[0] is None
            )
            self._retained.clear()
            self._closed = True
