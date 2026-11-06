# Owner(s): ["oncall: distributed"]

"""Opt-in two-process native tests. Requires a rebuilt TE and PyTorch 2.15."""

import asyncio
import faulthandler
import multiprocessing
import os
import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    parametrize,
    run_tests,
    TestCase,
)


# Cross TCP's default 64 KiB staging boundary, including a partial final chunk.
_NUMEL = 32769


def _receive(pipe):
    if not pipe.poll(30):
        raise TimeoutError("control-plane peer did not respond")
    return pipe.recv_bytes()


def _worker(rank, pipe, device_type, stream_mode):
    from torch.distributed._transport.mooncake import (
        MooncakeRemoteBuffer,
        MooncakeTransport,
    )

    checker = TestCase()
    faulthandler.dump_traceback_later(45)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    device = f"cuda:{rank}" if device_type == "cuda" else "cpu"
    with MooncakeTransport(
        device,
        host="127.0.0.1",
        protocol=os.getenv("MOONCAKE_TEST_PROTOCOL", "tcp"),
    ) as transport:
        checker.assertTrue(transport._managed, "requires managed Mooncake bindings")
        pipe.send_bytes(transport.bind())
        transport.connect(_receive(pipe))
        source = torch.arange(_NUMEL, dtype=torch.float32, device=device) + rank
        target = torch.zeros_like(source)
        if device_type == "cuda":
            torch.cuda.synchronize()
        local = transport.register_memory(source)
        output = transport.register_memory(target)
        pipe.send_bytes(output.to_remote_buffer().serialize())
        remote = MooncakeRemoteBuffer.deserialize(_receive(pipe))
        # Warm native connections outside capture, with initialized buffers.
        transport.write(local.to_view(), remote, async_op=True).wait()
        pipe.send_bytes(b"written")
        checker.assertEqual(_receive(pipe), b"written")
        checker.assertEqual(
            target, torch.arange(_NUMEL, device=device) + 1 - rank, exact_dtype=False
        )
        asyncio.run(transport.read_async(local.to_mutable_view(), remote))
        checker.assertEqual(
            source, torch.arange(_NUMEL, device=device) + rank, exact_dtype=False
        )
        pipe.send_bytes(b"finished")
        checker.assertEqual(_receive(pipe), b"finished")
        if stream_mode:
            stream = torch.cuda.Stream()
            if stream_mode == "graph":
                with torch.cuda.stream(stream):
                    transport.write_stream(local.to_view(), remote)
                stream.synchronize()
                pipe.send_bytes(b"stream-warmed")
                checker.assertEqual(_receive(pipe), b"stream-warmed")
                source.fill_(rank + 10)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    source.add_(1)
                    transport.write_stream(local.to_view(), remote)
                for replay in range(3):
                    # Replay on the default stream deliberately: native staging
                    # must not enqueue copies behind the graph's completion wait.
                    graph.replay()
                    torch.cuda.synchronize()
                    pipe.send_bytes(b"replayed")
                    checker.assertEqual(_receive(pipe), b"replayed")
                    checker.assertEqual(
                        target, torch.full_like(target, 12 - rank + replay)
                    )
                    pipe.send_bytes(b"checked")
                    checker.assertEqual(_receive(pipe), b"checked")
            else:
                with torch.cuda.stream(stream):
                    source.fill_(rank + 10)
                    transport.write_stream(local.to_view(), remote)
                stream.synchronize()
            pipe.send_bytes(b"stream-written")
            checker.assertEqual(_receive(pipe), b"stream-written")
            increment = 3 if stream_mode == "graph" else 0
            checker.assertEqual(target, torch.full_like(target, 11 - rank + increment))

            # Exercise read ordering too: the GPU consumer must see the remote
            # payload after a preceding kernel clears its local destination.
            def read_and_consume():
                source.zero_()
                transport.read_stream(local.to_mutable_view(), remote)
                source.add_(2)

            with torch.cuda.stream(stream):
                read_and_consume()
            stream.synchronize()
            checker.assertEqual(source, torch.full_like(source, rank + 12 + increment))
            if stream_mode == "graph":
                read_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(read_graph, stream=stream):
                    read_and_consume()
                for _ in range(3):
                    read_graph.replay()
                    torch.cuda.synchronize()
                    checker.assertEqual(
                        source, torch.full_like(source, rank + 12 + increment)
                    )
            # Captured registrations stay retained until close. No later replay
            # or remote access is allowed after the final handshake.
            pipe.send_bytes(b"stream-finished")
            checker.assertEqual(_receive(pipe), b"stream-finished")
        else:
            transport.unregister_memory(output)
            output = transport.register_memory(target)
            pipe.send_bytes(output.to_remote_buffer().serialize())
            remote = MooncakeRemoteBuffer.deserialize(_receive(pipe))
            transport.write(local.to_view(), remote)
            pipe.send_bytes(b"rewritten")
            checker.assertEqual(_receive(pipe), b"rewritten")
            checker.assertEqual(
                target,
                torch.arange(_NUMEL, device=device) + 1 - rank,
                exact_dtype=False,
            )
    pipe.close()
    faulthandler.cancel_dump_traceback_later()


def _multi_peer_worker(rank, connections):
    from torch.distributed._transport.mooncake import (
        MooncakeRemoteBuffer,
        MooncakeTransport,
    )

    checker = TestCase()
    # Rank 0 reads from ranks 1 and 2 through one endpoint.
    with MooncakeTransport(
        "cpu", host="127.0.0.1", protocol=os.getenv("MOONCAKE_TEST_PROTOCOL", "tcp")
    ) as transport:
        checker.assertTrue(transport._managed, "requires managed Mooncake bindings")
        tensor = torch.full((1024,), float(rank))
        memory = transport.register_memory(tensor)
        if rank == 0:
            urls = [_receive(connection) for connection in connections]
            for url in urls:
                transport.connect(url)
            remotes = [
                MooncakeRemoteBuffer.deserialize(_receive(c)) for c in connections
            ]
            targets = [torch.zeros(1024) for _ in remotes]
            views = [transport.register_memory(t).to_mutable_view() for t in targets]
            for _ in range(2):
                works = [
                    transport.read(view, remote, async_op=True)
                    for view, remote in zip(views, remotes)
                ]
                for work in works:
                    work.wait()
                for value, target in enumerate(targets, 1):
                    checker.assertEqual(target, torch.full_like(target, value))
                    target.zero_()
                # Reconnecting with the same bind() bytes is a no-op.
                transport.connect(urls[0])
        else:
            connections[0].send_bytes(transport.bind())
            connections[0].send_bytes(memory.to_remote_buffer().serialize())
        for connection in connections:
            connection.send_bytes(b"done")
            if _receive(connection) != b"done":
                raise AssertionError("unexpected control-plane message")
        transport.unregister_memory(memory)
        asyncio.run(transport.close_async(timeout=5))
    for connection in connections:
        connection.close()


def _run_workers(test, device_type, stream_mode=None):
    if device_type == "cuda" and torch.cuda.device_count() < 2:
        test.skipTest("requires two CUDA devices")
    # Fresh processes isolate native engine registration and graph callback state.
    context = multiprocessing.get_context("spawn")
    pipes = context.Pipe()
    workers = [
        context.Process(
            target=_worker, args=(rank, pipes[rank], device_type, stream_mode)
        )
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for pipe in pipes:
            pipe.close()
        for worker in workers:
            worker.join(60)
            test.assertEqual(worker.exitcode, 0)
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(10)


@unittest.skipUnless(
    os.getenv("MOONCAKE_TEST_TORCH_TRANSPORT") == "1",
    "set MOONCAKE_TEST_TORCH_TRANSPORT=1 with a rebuilt native TE",
)
class TestMooncakeNative(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_read_write(self, device):
        _run_workers(self, torch.device(device).type)


@unittest.skipUnless(
    os.getenv("MOONCAKE_TEST_TORCH_TRANSPORT") == "1",
    "set MOONCAKE_TEST_TORCH_TRANSPORT=1 with a rebuilt native TE",
)
class TestMooncakeNativeCUDAStream(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize("stream_mode", ("stream", "graph"))
    def test_read_write(self, device, stream_mode):
        _run_workers(self, torch.device(device).type, stream_mode)


@unittest.skipUnless(
    os.getenv("MOONCAKE_TEST_TORCH_TRANSPORT") == "1",
    "set MOONCAKE_TEST_TORCH_TRANSPORT=1 with a rebuilt native TE",
)
class TestMooncakeNativeCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_read_write(self, device):
        _run_workers(self, torch.device(device).type)


instantiate_device_type_tests(TestMooncakeNativeCUDA, globals(), only_for="cuda")
instantiate_device_type_tests(TestMooncakeNative, globals(), only_for="cpu")
instantiate_device_type_tests(TestMooncakeNativeCUDAStream, globals(), only_for="cuda")


@unittest.skipUnless(
    os.getenv("MOONCAKE_TEST_TORCH_TRANSPORT") == "1",
    "set MOONCAKE_TEST_TORCH_TRANSPORT=1 with a rebuilt native TE",
)
class TestMooncakeNativeMultiPeer(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_multi_peer(self, device):
        context = multiprocessing.get_context("spawn")
        pairs = [context.Pipe() for _ in range(2)]
        connections = [[pair[0] for pair in pairs], [pairs[0][1]], [pairs[1][1]]]
        workers = [
            context.Process(target=_multi_peer_worker, args=(rank, connection))
            for rank, connection in enumerate(connections)
        ]
        try:
            for worker in workers:
                worker.start()
            for pair in pairs:
                for connection in pair:
                    connection.close()
            for worker in workers:
                worker.join(60)
                self.assertEqual(worker.exitcode, 0)
        finally:
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
                if worker.pid is not None:
                    worker.join(10)


instantiate_device_type_tests(TestMooncakeNativeMultiPeer, globals(), only_for="cpu")

if __name__ == "__main__":
    run_tests()
