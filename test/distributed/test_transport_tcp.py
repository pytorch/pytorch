# Owner(s): ["oncall: distributed"]

import gc
import pickle
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from transport_test_utils import TransportTestMixin

import torch
from torch.distributed._transport._tcp import TCPRemoteBuffer, TCPTransport
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTCPTransport(TransportTestMixin, TestCase):
    def connect(self, flows: int = 3, chunk_size: int = 257, timeout: float = 5.0):
        server = TCPTransport(
            "cpu",
            num_flows=flows,
            host="127.0.0.1",
            chunk_size=chunk_size,
            timeout=timeout,
        )
        client = TCPTransport(
            "cpu",
            num_flows=flows,
            host="127.0.0.1",
            chunk_size=chunk_size,
            timeout=timeout,
        )
        self.assertEqual(client.connect(server.bind()), 0)
        deadline = time.monotonic() + 5
        while not server.connected() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(server.connected())
        self.assertTrue(client.connected())
        self.assertEqual(len(server._flows), flows)
        self.assertEqual(len(client._flows), flows)
        return server, client

    def make_transport_pair(self):
        return self.connect()

    def test_default_host_is_loopback_and_wildcard_is_rejected(self) -> None:
        transport = TCPTransport("cpu")
        try:
            self.assertEqual(transport._parse_url(transport.bind())[0], "127.0.0.1")
        finally:
            transport.close()
        with self.assertRaisesRegex(ValueError, "address peers can connect to"):
            TCPTransport("cpu", host="0.0.0.0")

    def test_close_releases_registered_tensors(self) -> None:
        transport = TCPTransport("cpu")
        tensor = torch.arange(8)
        memory = transport.register_memory(tensor)
        tensor_ref = weakref.ref(tensor)
        del tensor, memory

        self.assertIsNotNone(tensor_ref())
        transport.close()
        gc.collect()
        self.assertIsNone(tensor_ref())

    def test_read_write(self) -> None:
        server, client = self.connect()
        try:
            source = torch.arange(8192, dtype=torch.int32)
            destination = torch.zeros(4096, dtype=torch.int32)
            source_memory = client.register_memory(source)
            destination_memory = server.register_memory(destination)

            remote = destination_memory.to_remote_buffer()
            self.assertEqual(pickle.loads(pickle.dumps(remote)), remote)
            self.assertEqual(
                client.write(source_memory.to_view(4096 * 4, 4096 * 4), remote), 0
            )
            self.assertEqual(destination, source[4096:])

            result = torch.full((4100,), -1, dtype=torch.int32)
            result_memory = client.register_memory(result)
            self.assertEqual(
                client.read(
                    result_memory.to_mutable_view(8, destination.numel() * 4), remote
                ),
                0,
            )
            self.assertEqual(result[2:4098], destination)
            self.assertEqual(result[:2], torch.full((2,), -1, dtype=torch.int32))
            self.assertEqual(result[4098:], torch.full((2,), -1, dtype=torch.int32))
        finally:
            client.close()
            server.close()

    def test_bidirectional_write(self) -> None:
        server, client = self.connect(flows=4, chunk_size=1024)
        try:
            server_source = torch.arange(256, dtype=torch.uint8).repeat(256)
            client_source = server_source.bitwise_not()
            server_destination = torch.zeros_like(server_source)
            client_destination = torch.zeros_like(server_source)
            server_source_memory = server.register_memory(server_source)
            client_source_memory = client.register_memory(client_source)
            server_remote = server.register_memory(
                server_destination
            ).to_remote_buffer()
            client_remote = client.register_memory(
                client_destination
            ).to_remote_buffer()

            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [
                    pool.submit(
                        server.write,
                        server_source_memory.to_view(),
                        client_remote,
                    ),
                    pool.submit(
                        client.write,
                        client_source_memory.to_view(),
                        server_remote,
                    ),
                ]
                self.assertEqual([result.result() for result in results], [0, 0])
            self.assertEqual(client_destination, server_source)
            self.assertEqual(server_destination, client_source)
        finally:
            client.close()
            server.close()

    def test_symmetric_connect(self) -> None:
        first = TCPTransport("cpu", num_flows=2, host="127.0.0.1")
        second = TCPTransport("cpu", num_flows=2, host="127.0.0.1")
        try:
            first_url = first.bind()
            second_url = second.bind()
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = [
                    pool.submit(first.connect, second_url),
                    pool.submit(second.connect, first_url),
                ]
                self.assertEqual([result.result() for result in results], [0, 0])
            deadline = time.monotonic() + 5
            while (
                not first._passive_flows or not second._passive_flows
            ) and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(first._active_flows), 2)
            self.assertEqual(len(first._passive_flows), 2)
            self.assertEqual(len(second._active_flows), 2)
            self.assertEqual(len(second._passive_flows), 2)

            source = torch.arange(4096, dtype=torch.int32)
            destination = torch.zeros_like(source)
            source_memory = first.register_memory(source)
            destination_memory = second.register_memory(destination)
            self.assertEqual(
                first.write(
                    source_memory.to_view(), destination_memory.to_remote_buffer()
                ),
                0,
            )
            self.assertEqual(destination, source)
        finally:
            second.close()
            first.close()

    def test_registration_and_errors(self) -> None:
        server, client = self.connect()
        try:
            source = torch.arange(16, dtype=torch.uint8)
            destination = torch.zeros_like(source)
            source_memory = client.register_memory(source)
            first = server.register_memory(destination)
            second = server.register_memory(destination)
            self.assertFalse(first.reused_registration())
            self.assertTrue(second.reused_registration())
            self.assertEqual(first.to_remote_buffer(), second.to_remote_buffer())

            invalid = TCPRemoteBuffer(1, source.numel(), 2)
            with self.assertRaisesRegex(RuntimeError, "unknown remote buffer"):
                client.write(source_memory.to_view(), invalid)
            self.assertEqual(
                client.write(source_memory.to_view(), first.to_remote_buffer()), 0
            )
            self.assertEqual(destination, source)
            with self.assertRaisesRegex(ValueError, "outside"):
                source_memory.to_view(source.numel() + 1)
        finally:
            client.close()
            server.close()

    def test_operation_timeout(self) -> None:
        server, client = self.connect(timeout=0.05)
        try:
            tensor = torch.arange(8)
            source = client.register_memory(tensor)
            destination = server.register_memory(torch.zeros_like(tensor))
            with patch.object(client, "_queue_frame"):
                with self.assertRaisesRegex(TimeoutError, "timed out"):
                    client.write(source.to_view(), destination.to_remote_buffer())
            self.assertEqual(
                client.write(source.to_view(), destination.to_remote_buffer()), 0
            )
        finally:
            client.close()
            server.close()


if __name__ == "__main__":
    run_tests()
