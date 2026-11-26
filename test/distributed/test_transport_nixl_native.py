# Owner(s): ["oncall: distributed"]

import asyncio
import multiprocessing
import os
import unittest

import torch
import torch.distributed as dist
from torch.distributed._transport import new_transport, wait_all
from torch.distributed._transport.nixl import NIXLRemoteBuffer
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _receive(connection):
    if not connection.poll(10):
        raise TimeoutError("peer did not send control-plane metadata")
    return connection.recv_bytes()


def _native_worker(rank, connection):
    with new_transport("nixl", "cpu", timeout=5) as transport:
        connection.send_bytes(transport.bind())
        transport.connect(_receive(connection))
        source = torch.arange(1024, dtype=torch.float32) + rank
        target = torch.zeros_like(source)
        other_target = torch.zeros_like(source)
        source_memory = transport.register_memory(source)
        target_memory = transport.register_memory(target)
        other_memory = transport.register_memory(other_target)
        for memory in (source_memory, target_memory, other_memory):
            connection.send_bytes(memory.to_remote_buffer().serialize())
        remote_source, remote_target, remote_other = (
            NIXLRemoteBuffer.deserialize(_receive(connection)) for _ in range(3)
        )
        work = transport.write(source_memory.to_view(), remote_target, async_op=True)
        other_work = transport.write(
            source_memory.to_view(), remote_other, async_op=True
        )
        asyncio.run(wait_all([work, other_work], timeout=5))
        connection.send_bytes(b"written")
        if _receive(connection) != b"written":
            raise AssertionError("unexpected control-plane message")
        torch.testing.assert_close(
            target, torch.arange(1024, dtype=torch.float32) + 1 - rank
        )
        torch.testing.assert_close(other_target, target)
        asyncio.run(
            transport.read_async(
                target_memory.to_mutable_view(), remote_source, timeout=5
            )
        )
        torch.testing.assert_close(
            target, torch.arange(1024, dtype=torch.float32) + 1 - rank
        )
        part = NIXLRemoteBuffer.deserialize(remote_source.slice(400, 800).serialize())
        transport.read(other_memory.to_mutable_view(0, 800), part)
        torch.testing.assert_close(other_target[:200], target[100:300])
        connection.send_bytes(b"finished")
        if _receive(connection) != b"finished":
            raise AssertionError("unexpected control-plane message")
        for memory in (source_memory, target_memory, other_memory):
            transport.unregister_memory(memory)
        old_registrations = (source_memory._registration, target_memory._registration)
        source.add_(10)
        target.zero_()
        source_memory = transport.register_memory(source)
        target_memory = transport.register_memory(target)
        if any(
            memory._registration is old
            for memory, old in zip((source_memory, target_memory), old_registrations)
        ):
            raise AssertionError("unregistered allocation was reused")
        connection.send_bytes(target_memory.to_remote_buffer().serialize())
        remote_target = NIXLRemoteBuffer.deserialize(_receive(connection))
        transport.write(source_memory.to_view(), remote_target)
        connection.send_bytes(b"rewritten")
        if _receive(connection) != b"rewritten":
            raise AssertionError("unexpected control-plane message")
        torch.testing.assert_close(
            target, torch.arange(1024, dtype=torch.float32) + 11 - rank
        )
        transport.unregister_memory(source_memory)
        transport.unregister_memory(target_memory)
        asyncio.run(transport.close_async(timeout=5))
    connection.close()


def _mixed_device_worker(rank, connection, dtype):
    device = torch.device("cpu" if rank == 0 else "cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    payload = torch.arange(257, dtype=dtype)
    expected_source = torch.full((265,), -1, dtype=dtype)
    expected_source[3:260] = payload + rank
    source = expected_source.to(device, copy=True)
    target = torch.full((269,), -2, dtype=dtype, device=device)
    expected_target = torch.full((269,), -2, dtype=dtype)
    expected_target[5:262] = payload + 1 - rank
    itemsize = source.element_size()
    length = payload.numel() * itemsize
    with new_transport("nixl", device, timeout=5) as transport:
        connection.send_bytes(transport.bind())
        transport.connect(_receive(connection))
        source_memory = transport.register_memory(source)
        target_memory = transport.register_memory(target)
        for memory, offset in ((source_memory, 3), (target_memory, 5)):
            remote = memory.to_remote_buffer()
            connection.send_bytes(remote.slice(offset * itemsize, length).serialize())
        remote_source, remote_target = (
            NIXLRemoteBuffer.deserialize(_receive(connection)) for _ in range(2)
        )
        for operation in ("write", "read"):
            target.fill_(-2)
            # Ordinary transfers require the caller to finish CUDA producers.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            connection.send_bytes(b"ready")
            if _receive(connection) != b"ready":
                raise AssertionError("unexpected control-plane message")
            if operation == "write":
                work = transport.write(
                    source_memory.to_view(3 * itemsize, length),
                    remote_target,
                    async_op=True,
                )
            else:
                work = transport.read(
                    target_memory.to_mutable_view(5 * itemsize, length),
                    remote_source,
                    async_op=True,
                )
            work.wait()
            connection.send_bytes(b"finished")
            if _receive(connection) != b"finished":
                raise AssertionError("unexpected control-plane message")
            torch.testing.assert_close(target.cpu(), expected_target, rtol=0, atol=0)
            torch.testing.assert_close(source.cpu(), expected_source, rtol=0, atol=0)
    connection.close()


def _multi_peer_worker(rank, connections):
    # Rank 0 reads from ranks 1 and 2 through one endpoint.
    with new_transport("nixl", "cpu", timeout=5) as transport:
        tensor = torch.full((1024,), float(rank))
        memory = transport.register_memory(tensor)
        if rank == 0:
            urls = [_receive(connection) for connection in connections]
            for url in urls:
                transport.connect(url)
            remotes = [NIXLRemoteBuffer.deserialize(_receive(c)) for c in connections]
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
                    torch.testing.assert_close(target, torch.full_like(target, value))
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


def _passive_worker(rank, connection, operation):
    # Rank 1 only waits on the pipe; the progress thread must service rank 0's
    # one-sided transfer over TCP.
    with new_transport(
        "nixl", "cpu", timeout=5, backend_options={"engine_config": "TLS=tcp"}
    ) as transport:
        expected = torch.arange(4096, dtype=torch.int64)
        source_rank = 0 if operation == "write" else 1
        tensor = expected.clone() if rank == source_rank else torch.zeros_like(expected)
        memory = transport.register_memory(tensor)
        if rank == 1:
            connection.send_bytes(transport.bind())
            connection.send_bytes(memory.to_remote_buffer().serialize())
            if _receive(connection) != b"done":
                raise AssertionError("unexpected control-plane message")
        else:
            transport.connect(_receive(connection))
            remote = NIXLRemoteBuffer.deserialize(_receive(connection))
            if operation == "write":
                transport.write(memory.to_view(), remote)
            else:
                transport.read(memory.to_mutable_view(), remote)
            connection.send_bytes(b"done")
        torch.testing.assert_close(tensor, expected)
    connection.close()


def _connect_rank_worker(rank, port):
    # Rank 0 connects to ranks 1 and 2 with connect_rank and reads from both.
    store = dist.TCPStore("127.0.0.1", port, is_master=False)
    with new_transport("nixl", "cpu", timeout=5) as transport:
        memory = transport.register_memory(torch.full((1024,), float(rank)))
        if rank == 0:
            for peer in (1, 2):
                transport.connect_rank(
                    peer, store=store, rank=0, tag="nixl", timeout=10
                )
            for peer in (1, 2):
                remote = NIXLRemoteBuffer.deserialize(store.get(f"remote/{peer}"))
                target = torch.zeros(1024)
                view = transport.register_memory(target).to_mutable_view()
                transport.read(view, remote)
                torch.testing.assert_close(target, torch.full_like(target, peer))
            store.set("done", "1")
        else:
            transport.connect_rank(0, store=store, rank=rank, tag="nixl", timeout=10)
            # pyrefly: ignore [bad-argument-type]
            store.set(f"remote/{rank}", memory.to_remote_buffer().serialize())
            store.wait(["done"])
        asyncio.run(transport.close_async(timeout=5))


def _next_notifications(transport):
    notifications = transport.wait_notifications(timeout=10)
    if not notifications:
        raise AssertionError("timed out waiting for a notification")
    return notifications


def _notification_worker(rank, connection, progress_thread):
    with new_transport(
        "nixl", "cpu", timeout=10, enable_prog_thread=progress_thread
    ) as transport:
        connection.send_bytes(transport.bind())
        transport.connect(_receive(connection))
        data = torch.full((1 << 20,), rank + 1, dtype=torch.uint8)
        memory = transport.register_memory(data)
        connection.send_bytes(memory.to_remote_buffer().serialize())
        remote = NIXLRemoteBuffer.deserialize(_receive(connection))
        if rank == 0:
            # Never waited on before the reply: the transport sends it itself.
            work = transport.write(
                memory.to_view(), remote, async_op=True, notification=b"written"
            )
            if _next_notifications(transport) != [(remote.peer_id, b"read")]:
                raise AssertionError("unexpected notification")
            work.wait()
            connection.send_bytes(b"done")
        else:
            if _next_notifications(transport) != [(remote.peer_id, b"written")]:
                raise AssertionError("unexpected notification")
            torch.testing.assert_close(data, torch.ones_like(data))
            transport.send_notification(remote.peer_id, b"read").wait()
            _receive(connection)  # Stay open until the reply arrives.


def _run_worker(worker, *args):
    # Some NIXL 0.6-1.5 wheels segfault at exit on hosts with libefa, whose UCX modules
    # load a second libnl-route-3: https://github.com/ai-dynamo/nixl/issues/2428
    os.environ.setdefault("UCX_MODULES", "^efa,rdmacm")
    worker(*args)


@instantiate_parametrized_tests
@unittest.skipUnless(
    os.environ.get("TORCH_TEST_NIXL") == "1",
    "set TORCH_TEST_NIXL=1 with NIXL/UCX installed",
)
class TestNIXLNative(TestCase):
    def test_two_process_cpu_transfers(self):
        connections = multiprocessing.Pipe()
        self._spawn(
            _native_worker, [(connection,) for connection in connections], connections
        )

    @parametrize("operation", ["read", "write"])
    def test_passive_peer_tcp(self, operation):
        connections = multiprocessing.Pipe()
        self._spawn(
            _passive_worker,
            [(connection, operation) for connection in connections],
            connections,
        )

    @parametrize("progress_thread", [True, False])
    def test_two_process_notifications(self, progress_thread):
        connections = multiprocessing.Pipe()
        self._spawn(
            _notification_worker,
            [(connection, progress_thread) for connection in connections],
            connections,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires one CUDA device")
    @parametrize("dtype", [torch.float32, torch.int64])
    def test_two_process_mixed_device_transfers(self, dtype):
        connections = multiprocessing.Pipe()
        self._spawn(
            _mixed_device_worker,
            [(connection, dtype) for connection in connections],
            connections,
        )

    def test_three_process_cpu_multi_peer(self):
        pipes = [multiprocessing.Pipe() for _ in range(2)]
        self._spawn(
            _multi_peer_worker,
            [([first for first, _ in pipes],), ([pipes[0][1]],), ([pipes[1][1]],)],
            [connection for pipe in pipes for connection in pipe],
        )

    def test_three_process_cpu_connect_rank(self):
        store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
        self._spawn(_connect_rank_worker, [(store.port,)] * 3, [])

    def _spawn(self, worker, args, connections):
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_run_worker, args=(worker, rank, *rank_args))
            for rank, rank_args in enumerate(args)
        ]
        try:
            for process in processes:
                process.start()
            for connection in connections:
                connection.close()
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join(5)


if __name__ == "__main__":
    run_tests()
