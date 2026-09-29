"""Private CUDA ordering prototype. Native CUDA calls run outside host callbacks."""

from __future__ import annotations

import ctypes
import os
import platform
import queue
import threading
from contextlib import contextmanager
from typing import Any, cast, TYPE_CHECKING

import torch

from ._memory import NIXLMemoryView, NIXLMutableMemoryView, NIXLRemoteBuffer
from ._work import _PollingWork


if TYPE_CHECKING:
    from ._transport import NIXLTransport


_LIVE_BRIDGES: set[_CudaStreamBridge] = set()


class _CudaWork(_PollingWork):
    def __init__(self, timeout, captured):
        super().__init__(timeout)
        self.done = threading.Event()
        self.captured = captured

    def _poll(self):
        if self.captured:
            raise RuntimeError(
                "captured Work has no per-replay host completion; synchronize the graph stream"
            )
        return self.done.is_set()


class _CudaStreamBridge:
    """Private per-stream callback lifetime owner for explicit CUDA stream transfers."""

    def __init__(self, transport: NIXLTransport, stream: torch.cuda.Stream):
        if transport._plugin != "UCX":
            raise ValueError("only the UCX CUDA path is supported")
        from cuda.bindings import driver

        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("CUDA gate prototype requires Linux/x86-64")
        self.driver = driver
        result = driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
            stream.device.index,
        )
        if int(result[0]) or result[1] < 100:
            raise RuntimeError(
                "CUDA gate requires native owner-device RDMA write ordering"
            )
        result = driver.cuMemHostAlloc(256, 2)
        if int(result[0]):
            raise RuntimeError(f"CUDA gate allocation failed: {result[0]}")
        self._host = result[1]
        result = driver.cuMemHostGetDevicePointer(self._host, 0)
        if int(result[0]):
            driver.cuMemFreeHost(self._host)
            raise RuntimeError(f"CUDA gate mapping failed: {result[0]}")
        self._device_address = int(result[1])
        self._free_slots = list(range(64))
        self.transport = transport
        self.stream = stream
        # Call the driver directly: binding-level one-shot callback userdata
        # must not be released after the first execution of a captured node.
        self._cuda = ctypes.CDLL("libcuda.so.1")
        self._launch = self._cuda.cuLaunchHostFunc
        self._launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self._launch.restype = ctypes.c_int
        self._lock = threading.Lock()
        self._closed = False
        self._pending: list[tuple[Any, ...]] = []
        self._graphs: list[torch.cuda.CUDAGraph] = []
        self._captured: list[tuple[Any, ...]] = []
        self._capturing = False
        self._queue = queue.Queue()
        self._ready = threading.Event()
        self._init_error = None
        self._thread = threading.Thread(target=self._progress, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._init_error is not None:
            driver.cuMemFreeHost(self._host)
            raise self._init_error
        _LIVE_BRIDGES.add(self)

    def _progress(self):
        try:
            torch.cuda.set_device(self.stream.device)
        except BaseException as error:
            self._init_error = error
            return
        finally:
            self._ready.set()
        while True:
            item = self._queue.get()
            if item is None:
                return
            transfer, flag = item
            transfer()
            flag.value = 1

    @contextmanager
    def capture(self):
        """Capture on the bridge stream; close invalidates all returned graphs.

        Register and warm up first. Replay serially, with fixed buffers and
        descriptors. Do not clone graphs, recapture externally, or race close
        with capture/replay. Use the replay stream for completion observation.
        """
        if self._closed or self._capturing:
            raise RuntimeError("bridge is closed or already capturing")
        graph = torch.cuda.CUDAGraph()
        self._graphs.append(graph)
        self._capturing = True
        try:
            with torch.cuda.graph(graph, stream=self.stream):
                yield graph
        finally:
            self._capturing = False

    def write(self, local: NIXLMemoryView, remote: NIXLRemoteBuffer, *, timeout=30.0):
        return self._enqueue("write", local, remote, timeout)

    def read(
        self, local: NIXLMutableMemoryView, remote: NIXLRemoteBuffer, *, timeout=30.0
    ):
        return self._enqueue("read", local, remote, timeout)

    def _enqueue(self, operation, local, remote, timeout) -> _CudaWork:
        from .._work import _validate_timeout

        _validate_timeout(timeout)
        if timeout is None or timeout == 0:
            raise ValueError("CUDA transfer timeout must be finite and positive")
        if not isinstance(local, NIXLMemoryView) or not isinstance(
            remote, NIXLRemoteBuffer
        ):
            raise TypeError("expected NIXL memory views/descriptors")
        if operation == "read" and not isinstance(local, NIXLMutableMemoryView):
            raise TypeError("read requires a mutable view")
        memory = local._memory
        memory._registration.ensure_active()
        tensor = memory._registration.tensor
        if memory._transport is not self.transport:
            raise ValueError("memory belongs to another transport")
        if not tensor.is_cuda:
            raise ValueError("CPU transfers must bypass CUDA ordering")
        self.transport._ensure_open()
        if self.transport._peer_name != remote.agent_name:
            raise ValueError("remote buffer does not belong to the connected peer")
        if local.size() > remote.length:
            raise ValueError("local view exceeds remote buffer")
        with (
            self._lock,
            torch.cuda.device(self.stream.device),
            torch.cuda.stream(self.stream),
        ):
            if self._closed:
                raise RuntimeError("bridge is closed")
            capturing = torch.cuda.is_current_stream_capturing()
            if capturing and not self._capturing:
                raise RuntimeError(
                    "use transport.cuda_graph() to retain graph callbacks"
                )
            if not capturing:
                pending = []
                for item in self._pending:
                    if item[0].query():
                        self._free_slots.append(item[5])
                    else:
                        pending.append(item)
                self._pending = pending
            if not self._free_slots:
                raise RuntimeError(
                    "CUDA gate capacity exhausted; synchronize before enqueueing more"
                )
            slot = self._free_slots.pop()
            flag = ctypes.c_uint32.from_address(int(self._host) + slot * 4)
            flag.value = 0
            work_result = _CudaWork(timeout, capturing)
            retired = None if capturing else torch.cuda.Event()

            def transfer():
                try:
                    # This thread, unlike CUDA's callback thread, may call CUDA.
                    work = self.transport._start(
                        operation.upper(),
                        local,
                        remote,
                        mutable=operation == "read",
                        async_op=True,
                        timeout=timeout,
                    )
                    cast(torch.distributed.Work, work).wait()
                    work_result.done.set()
                except BaseException as error:
                    os.write(2, f"CUDA transport failed: {error}\n".encode())
                    os._exit(1)

            @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
            def callback(_):
                try:
                    # A replay resets its own slot before the GPU reaches the wait.
                    flag.value = 0
                    self._queue.put((transfer, flag))
                except BaseException as error:
                    os.write(2, f"CUDA transport callback failed: {error}\n".encode())
                    os._exit(1)

            # Keep the ctypes trampoline and operands alive until CUDA confirms
            # the callback returned, not merely until done.set() executes.
            item = (retired, callback, local, remote, work_result, slot)
            pending = self._captured if capturing else self._pending
            pending.append(item)
            result = self._launch(
                self.stream.cuda_stream,
                ctypes.cast(callback, ctypes.c_void_p),
                None,
            )
            if result:
                pending.remove(item)
                self._free_slots.append(slot)
                raise RuntimeError(f"CUDA driver error: {result}")
            try:
                result = self.driver.cuStreamWaitValue32(
                    self.driver.CUstream(self.stream.cuda_stream),
                    self._device_address + slot * 4,
                    1,
                    0,
                )
                if int(result[0]):
                    raise RuntimeError(f"CUDA completion gate failed: {result[0]}")
                if retired is not None:
                    retired.record(self.stream)
            except BaseException as error:
                # A native submission is queued: never let consumers proceed
                # without a successfully enqueued completion dependency.
                os.write(2, f"CUDA gate failed: {error}\n".encode())
                os._exit(1)
            return work_result

    def owns(self, registration):
        return any(
            item[2]._memory._registration is registration for item in self._captured
        ) or any(
            item[2]._memory._registration is registration and not item[0].query()
            for item in self._pending
        )

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._capturing:
                raise RuntimeError("cannot close during capture")
            # Replays may run on a different stream. Keep callbacks alive until
            # all launches finish and graph executables have been destroyed.
            torch.cuda.synchronize(self.stream.device)
            for graph in self._graphs:
                graph.reset()
            self._graphs.clear()
            self._captured.clear()
            self._pending.clear()
            self._closed = True
            self._queue.put(None)
            self._thread.join()
            result = self.driver.cuMemFreeHost(self._host)
            if int(result[0]):
                raise RuntimeError(f"CUDA gate release failed: {result[0]}")
            _LIVE_BRIDGES.discard(self)
