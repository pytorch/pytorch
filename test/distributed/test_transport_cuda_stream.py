# Owner(s): ["oncall: distributed"]

import asyncio
import gc
import multiprocessing
import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.distributed._transport.nixl import NIXLTransport
from torch.distributed._transport.nixl._memory import NIXLRemoteBuffer
from torch.testing._internal.common_utils import run_tests, TestCase


def _worker(rank, pipe, capture=False):
    torch.cuda.set_device(rank)
    transport = NIXLTransport()
    stream = torch.cuda.Stream()
    source = torch.empty(1024, device="cuda")
    destination = torch.zeros(1024, device="cuda")
    producer = torch.full((1024,), rank + 1.0, device="cuda")
    consumer = torch.zeros_like(producer)
    # Prewarm kernels and memory copies before queuing host callbacks.
    source.copy_(producer)
    consumer.copy_(destination)
    torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
    source.zero_()
    local_source = transport.register_memory(source)
    local_destination = transport.register_memory(destination)
    pipe.send_bytes(transport.bind())
    transport.connect(pipe.recv_bytes())
    for memory in (local_source, local_destination):
        pipe.send_bytes(memory.to_remote_buffer().serialize())
    remote_source = NIXLRemoteBuffer.deserialize(pipe.recv_bytes())
    remote_destination = NIXLRemoteBuffer.deserialize(pipe.recv_bytes())
    if capture:
        # Initialize native UCX CUDA resources outside graph execution.
        with torch.cuda.stream(stream):
            transport.write_stream(local_source.to_view(), remote_destination)
        stream.synchronize()
        pipe.send_bytes(b"warm")
        if pipe.recv_bytes() != b"warm":
            raise AssertionError("warmup rendezvous")
        with torch.cuda.stream(stream):
            transport.read_stream(local_destination.to_mutable_view(), remote_source)
        stream.synchronize()
        write_graph, read_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(write_graph):
            torch.cuda._sleep(1_000_000)
            source.copy_(producer, non_blocking=True)
            transport.write_stream(local_source.to_view(), remote_destination)
        with torch.cuda.graph(read_graph):
            transport.read_stream(local_destination.to_mutable_view(), remote_source)
            consumer.copy_(destination, non_blocking=True)
        torch.testing.assert_close(source, torch.zeros_like(source))
        torch.testing.assert_close(destination, torch.zeros_like(destination))
        pipe.send_bytes(b"captured")
        if not (pipe.recv_bytes() == b"captured"):
            raise AssertionError("unexpected capture result or peer message")
        try:
            transport.unregister_memory(local_source)
        except RuntimeError:
            pass
        else:
            raise AssertionError("graph registration was unregistered")
        for iteration in range(3):
            producer.fill_(rank + 1.0 + 10 * iteration)
            # Replay on the default stream, not the original capture stream.
            write_graph.replay()
            torch.cuda.synchronize()
            pipe.send_bytes(b"written")
            if not (pipe.recv_bytes() == b"written"):
                raise AssertionError("unexpected capture result or peer message")
            read_graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                consumer, torch.full_like(consumer, 2 - rank + 10 * iteration)
            )
            # Eager completion/reaping and GC must not free graph callbacks.
            transport.read_stream(local_destination.to_mutable_view(), remote_source)
            torch.cuda.synchronize()
            gc.collect()
            pipe.send_bytes(b"iteration")
            if not (pipe.recv_bytes() == b"iteration"):
                raise AssertionError("unexpected capture result or peer message")
    else:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1_000_000)
            source.copy_(producer, non_blocking=True)
            transport.write_stream(local_source.to_view(), remote_destination)
        stream.synchronize()
        # Local write completion does not imply peer write completion.
        pipe.send_bytes(b"written")
        if pipe.recv_bytes() != b"written":
            raise RuntimeError("unexpected control message")
        torch.testing.assert_close(destination, torch.full_like(destination, 2 - rank))
        with torch.cuda.stream(stream):
            transport.read_stream(local_destination.to_mutable_view(), remote_source)
            consumer.copy_(destination, non_blocking=True)
        stream.synchronize()  # Verify results only; no synchronization in enqueue.
        torch.testing.assert_close(consumer, torch.full_like(consumer, 2 - rank))
    if not capture:
        asyncio.run(
            transport.read_async(local_destination.to_mutable_view(), remote_source)
        )
        torch.testing.assert_close(destination, torch.full_like(destination, 2 - rank))
    # Stop peer accesses before closing local registrations.
    pipe.send_bytes(b"finished")
    pipe.recv_bytes()
    orderings = list(transport._cuda_streams.values())
    transport.close()
    if any(o._retained for o in orderings):
        raise AssertionError("graph resources retained after close")


def _callback_worker(fail, capture=False):
    torch.cuda.set_device(0)
    transport = NIXLTransport()
    tensor = torch.zeros(8, device="cuda")
    memory = transport.register_memory(tensor)
    remote = memory.to_remote_buffer()
    transport._peer_name = remote.agent_name
    stream = torch.cuda.Stream()
    if not fail:
        native_work = SimpleNamespace(wait=lambda: True)
        with (
            patch.object(transport, "_submit", return_value=native_work),
            patch.object(
                transport,
                "_cuda_stream",
                side_effect=AssertionError("ordinary API used stream ordering"),
            ),
        ):
            if transport.write(memory.to_view(), remote) != 0:
                raise AssertionError("ordinary synchronous write changed")
            if (
                transport.read(memory.to_mutable_view(), remote, async_op=True)
                is not native_work
            ):
                raise AssertionError("ordinary asynchronous read changed")
    entered, release = threading.Event(), threading.Event()

    async def transfer(*args, **kwargs):
        entered.set()
        if fail:
            raise RuntimeError("injected callback failure")
        if not await asyncio.to_thread(release.wait, 5):
            raise TimeoutError("enqueue blocked the submitting thread")

    with (
        patch.object(transport, "_transfer_async", side_effect=transfer),
        torch.cuda.stream(stream),
    ):
        if capture:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                transport.write_stream(memory.to_view(), remote)
            graph.replay()
        else:
            transport.write_stream(memory.to_view(), remote)
        done = torch.cuda.Event()
        done.record()
        if fail:
            torch.cuda.synchronize()  # Includes graph replays on the default stream.
            raise AssertionError("failed callback returned successfully")
        if not entered.wait(5) or done.query():
            raise AssertionError("expected pending callback after enqueue returned")
        try:
            transport.unregister_memory(memory)
        except RuntimeError:
            pass
        else:
            raise AssertionError("queued CUDA memory was unregistered")
        release.set()
        transport.close()
        if not done.query():
            raise AssertionError("missing callback completion")
    transport.close()


@unittest.skipUnless(
    os.getenv("TORCH_TEST_CUDA_TRANSPORT") == "1" and torch.cuda.device_count() >= 2,
    "requires TORCH_TEST_CUDA_TRANSPORT=1 and two GPUs",
)
class TestCudaStreamTransport(TestCase):
    def test_enqueue_returns_before_completion(self):
        self._check_callback_worker(False, 0)

    def test_callback_failure_terminates_process(self):
        self._check_callback_worker(True, 1)

    def test_graph_callback_failure_terminates_process(self):
        self._check_callback_worker(True, 1, capture=True)

    def _check_callback_worker(self, fail, exitcode, capture=False):
        process = multiprocessing.get_context("spawn").Process(
            target=_callback_worker, args=(fail, capture)
        )
        process.start()
        try:
            process.join(30)
            self.assertEqual(process.exitcode, exitcode)
        finally:
            if process.is_alive():
                process.kill()
            process.join()

    def test_native_stream_write_and_read(self):
        self._check_native(False)

    def test_native_cuda_graph_replay(self):
        self._check_native(True)

    def _check_native(self, capture):
        ctx = multiprocessing.get_context("spawn")
        pipes = ctx.Pipe()
        processes = [
            ctx.Process(target=_worker, args=(rank, pipes[rank], capture))
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(40)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.kill()
                process.join()


if __name__ == "__main__":
    run_tests()
