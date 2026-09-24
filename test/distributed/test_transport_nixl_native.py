# Owner(s): ["oncall: distributed"]

import asyncio
import multiprocessing
import os
import unittest

import torch
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


def _native_worker(rank, connection, progress_thread):
    with new_transport(
        "nixl", "cpu", timeout=5, enable_prog_thread=progress_thread
    ) as transport:
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
        connection.send_bytes(b"finished")
        if _receive(connection) != b"finished":
            raise AssertionError("unexpected control-plane message")
        for memory in (source_memory, target_memory, other_memory):
            transport.unregister_memory(memory)
        source.add_(10)
        target.zero_()
        source_memory = transport.register_memory(source)
        target_memory = transport.register_memory(target)
        if source_memory.reused_registration() or target_memory.reused_registration():
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


if __name__ == "__main__":
    run_tests()
