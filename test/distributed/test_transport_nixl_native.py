# Owner(s): ["oncall: distributed"]

import asyncio
import multiprocessing
import os
import time
import unittest
from dataclasses import replace

import torch
from torch.distributed._transport import new_transport, wait_all
from torch.distributed._transport.nixl import NIXLRemoteBuffer
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _receive(connection, transport):
    deadline = time.monotonic() + 10
    while not connection.poll(0.001):
        # Without a progress thread, UCX also needs polling at the receiver.
        transport._agent.get_new_notifs()
        if time.monotonic() >= deadline:
            raise TimeoutError("peer did not send control-plane metadata")
    return connection.recv_bytes()


def _native_worker(rank, connection, progress_thread):
    with new_transport(
        "nixl", "cpu", timeout=5, enable_prog_thread=progress_thread
    ) as transport:
        connection.send_bytes(transport.bind())
        transport.connect(_receive(connection, transport))
        source = torch.arange(1024, dtype=torch.float32) + rank
        target = torch.zeros_like(source)
        other_target = torch.zeros_like(source)
        source_memory = transport.register_memory(source)
        target_memory = transport.register_memory(target)
        other_memory = transport.register_memory(other_target)
        for memory in (source_memory, target_memory, other_memory):
            connection.send_bytes(memory.to_remote_buffer().serialize())
        remote_source, remote_target, remote_other = (
            NIXLRemoteBuffer.deserialize(_receive(connection, transport))
            for _ in range(3)
        )
        work = transport.write(source_memory.to_view(), remote_target, async_op=True)
        other_work = transport.write(
            source_memory.to_view(), remote_other, async_op=True
        )
        asyncio.run(wait_all([work, other_work], timeout=5))
        connection.send_bytes(b"written")
        if _receive(connection, transport) != b"written":
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
        connection.send_bytes(b"finished")
        if _receive(connection, transport) != b"finished":
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
        remote_target = NIXLRemoteBuffer.deserialize(_receive(connection, transport))
        transport.write(source_memory.to_view(), remote_target)
        connection.send_bytes(b"rewritten")
        if _receive(connection, transport) != b"rewritten":
            raise AssertionError("unexpected control-plane message")
        torch.testing.assert_close(
            target, torch.arange(1024, dtype=torch.float32) + 11 - rank
        )
        transport.unregister_memory(source_memory)
        transport.unregister_memory(target_memory)
        asyncio.run(transport.close_async(timeout=5))
    connection.close()


def _mixed_device_worker(rank, connection, device, dtype, progress_thread):
    device = torch.device("cpu" if rank == 0 else device)
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
    with new_transport(
        "nixl", device, timeout=5, enable_prog_thread=progress_thread
    ) as transport:
        connection.send_bytes(transport.bind())
        transport.connect(_receive(connection, transport))
        source_memory = transport.register_memory(source)
        target_memory = transport.register_memory(target)
        for memory, offset in ((source_memory, 3), (target_memory, 5)):
            remote = memory.to_remote_buffer()
            connection.send_bytes(
                replace(
                    remote, address=remote.address + offset * itemsize, length=length
                ).serialize()
            )
        remote_source, remote_target = (
            NIXLRemoteBuffer.deserialize(_receive(connection, transport))
            for _ in range(2)
        )
        for operation in ("write", "read"):
            target.fill_(-2)
            # Ordinary transfers require the caller to finish CUDA producers.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            connection.send_bytes(b"ready")
            if _receive(connection, transport) != b"ready":
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
            if _receive(connection, transport) != b"finished":
                raise AssertionError("unexpected control-plane message")
            torch.testing.assert_close(target.cpu(), expected_target, rtol=0, atol=0)
            torch.testing.assert_close(source.cpu(), expected_source, rtol=0, atol=0)
    connection.close()


@instantiate_parametrized_tests
@unittest.skipUnless(
    os.environ.get("TORCH_TEST_NIXL") == "1",
    "set TORCH_TEST_NIXL=1 with NIXL/UCX installed",
)
class TestNIXLNative(TestCase):
    @parametrize("progress_thread", [True, False])
    def test_two_process_cpu_transfers(self, progress_thread):
        context = multiprocessing.get_context("spawn")
        connections = context.Pipe()
        processes = [
            context.Process(
                target=_native_worker, args=(rank, connections[rank], progress_thread)
            )
            for rank in range(2)
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


@unittest.skipUnless(
    os.environ.get("TORCH_TEST_NIXL") == "1",
    "set TORCH_TEST_NIXL=1 with NIXL/UCX installed",
)
class TestNIXLMixedDevice(TestCase):
    @dtypes(torch.float32, torch.int64)
    @parametrize("progress_thread", [True, False])
    def test_two_process_transfers(self, device, dtype, progress_thread):
        context = multiprocessing.get_context("spawn")
        connections = context.Pipe()
        processes = [
            context.Process(
                target=_mixed_device_worker,
                args=(rank, connections[rank], device, dtype, progress_thread),
            )
            for rank in range(2)
        ]
        try:
            for process in processes:
                process.start()
            for connection in connections:
                connection.close()
            for process in processes:
                process.join(60)
                self.assertEqual(process.exitcode, 0)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join(5)


instantiate_device_type_tests(TestNIXLMixedDevice, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
