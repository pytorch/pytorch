# Owner(s): ["oncall: distributed"]

import multiprocessing
import os
import unittest

import torch
from torch.distributed._transport import new_transport
from torch.testing._internal.common_utils import run_tests, TestCase


def _receive(connection):
    if not connection.poll(10):
        raise TimeoutError("peer did not send control-plane metadata")
    return connection.recv()


def _native_worker(rank, connection):
    with new_transport("nixl", "cpu", timeout=5) as transport:
        connection.send(transport.bind())
        transport.connect(_receive(connection))
        source = torch.arange(1024, dtype=torch.float32) + rank
        target = torch.zeros_like(source)
        source_memory = transport.register_memory(source)
        target_memory = transport.register_memory(target)
        connection.send(
            (source_memory.to_remote_buffer(), target_memory.to_remote_buffer())
        )
        remote_source, remote_target = _receive(connection)
        work = transport.write(source_memory.to_view(), remote_target, async_op=True)
        work.wait()
        connection.send("written")
        if _receive(connection) != "written":
            raise AssertionError("unexpected control-plane message")
        torch.testing.assert_close(
            target, torch.arange(1024, dtype=torch.float32) + 1 - rank
        )
        transport.read(target_memory.to_mutable_view(), remote_source, timeout=5)
        torch.testing.assert_close(
            target, torch.arange(1024, dtype=torch.float32) + 1 - rank
        )
        connection.send("finished")
        if _receive(connection) != "finished":
            raise AssertionError("unexpected control-plane message")
    connection.close()


@unittest.skipUnless(
    os.environ.get("TORCH_TEST_NIXL") == "1",
    "set TORCH_TEST_NIXL=1 with NIXL/UCX installed",
)
class TestNIXLNative(TestCase):
    def test_two_process_cpu_transfers(self):
        context = multiprocessing.get_context("spawn")
        connections = context.Pipe()
        processes = [
            context.Process(target=_native_worker, args=(rank, connections[rank]))
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
