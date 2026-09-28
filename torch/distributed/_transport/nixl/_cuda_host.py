"""Opt-in CUDA host-callback prototype for NIXL pinned-host staging.

Direct VRAM transfers are rejected: NIXL/UCX can call CUDA APIs, which are
forbidden in CUDA host callbacks. See the transport CUDA stream design.
"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, TYPE_CHECKING

import torch

from ._memory import NIXLMemoryView, NIXLMutableMemoryView, NIXLRemoteBuffer


if TYPE_CHECKING:
    from ._transport import NIXLTransport


_LIVE_BRIDGES: set[CudaHostTransport] = set()


class CudaHostTransport:
    """Experimental stream-ordered transport between pinned CPU buffers.

    CUDA copies before/after the callback provide GPU staging. Only the UCX
    plugin is accepted. The caller must register and connect beforehand, retain
    the bridge, and close it before unregistering or closing the transport.
    Construction/close may synchronize; enqueue does not explicitly synchronize.
    Callback failures terminate the process, matching Mooncake's error policy.
    """

    def __init__(self, transport: NIXLTransport, stream: torch.cuda.Stream):
        if transport._plugin != "UCX":
            raise ValueError("only the UCX pinned-host path is supported")
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
        _LIVE_BRIDGES.add(self)

    @contextmanager
    def capture(self):
        """Capture on the bridge stream; close invalidates all returned graphs.

        Register and warm up first. Replay serially, with fixed buffers and
        descriptors. Do not clone graphs, recapture externally, or race close
        with capture/replay. Captured read/write return None, not a one-shot event.
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

    def _enqueue(self, operation, local, remote, timeout) -> threading.Event | None:
        from .._work import _validate_timeout

        _validate_timeout(timeout)
        if timeout is None:
            raise ValueError("callback timeout must be finite")
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
        if tensor.is_cuda or remote.memory_type != "DRAM":
            raise ValueError("CUDA callbacks cannot submit direct VRAM transfers")
        if not tensor.is_pinned():
            raise ValueError("local CPU memory must be pinned for CUDA staging")
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
                raise RuntimeError("use bridge.capture() to retain graph callbacks")
            if not capturing:
                self._pending = [item for item in self._pending if not item[0].query()]
            done = None if capturing else threading.Event()
            retired = None if capturing else torch.cuda.Event()

            @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
            def callback(_):
                try:
                    deadline = time.monotonic() + timeout
                    work = getattr(self.transport, operation)(
                        local, remote, async_op=True, timeout=timeout
                    )
                    while not work.is_completed():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("CUDA callback transport timed out")
                        time.sleep(0.001)
                    work.wait()  # Complete: read a terminal error, no pending wait.
                    if done is not None:
                        done.set()  # No user Future callbacks on this thread.
                except BaseException as error:
                    os.write(2, f"CUDA transport callback failed: {error}\n".encode())
                    os._exit(1)  # Never run GPU consumers after failed DMA.

            # Keep the ctypes trampoline and operands alive until CUDA confirms
            # the callback returned, not merely until done.set() executes.
            item = (retired, callback, local, remote, done)
            pending = self._captured if capturing else self._pending
            pending.append(item)
            result = self._launch(
                self.stream.cuda_stream,
                ctypes.cast(callback, ctypes.c_void_p),
                None,
            )
            if result:
                pending.remove(item)
                raise RuntimeError(f"CUDA driver error: {result}")
            try:
                if retired is not None:
                    retired.record(self.stream)
            except BaseException:
                # The callback may already be running: retain its trampoline and
                # reject further use rather than reclaim it from an unrecorded event.
                self._closed = True
                raise
            return done

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
            _LIVE_BRIDGES.discard(self)
