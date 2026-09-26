import asyncio

import torch
from torch.distributed._transport import MemoryView, MutableMemoryView, wait_all, Work


class TransportTestMixin:
    def make_transport_pair(self):
        raise NotImplementedError

    def test_transport_conformance(self) -> None:
        first, second = self.make_transport_pair()
        try:
            source = torch.arange(4096, dtype=torch.int32)
            destination = torch.zeros_like(source)
            source_memory = first.register_memory(source)
            destination_memory = second.register_memory(destination)
            descriptor = destination_memory.to_remote_buffer()
            remote = type(descriptor).deserialize(descriptor.serialize())

            self.assertIsInstance(source_memory.to_view(), MemoryView)
            self.assertNotIsInstance(source_memory.to_view(), MutableMemoryView)
            self.assertIsInstance(
                destination_memory.to_mutable_view(), MutableMemoryView
            )
            self.assertTrue(destination_memory.to_mutable_view().writable)
            self.assertEqual(source_memory.to_view().size(), source.nbytes)
            self.assertEqual(destination_memory.to_mutable_view(16, 32).size(), 32)
            self.assertEqual(first.write(source_memory.to_view(), remote), 0)
            self.assertEqual(destination, source)

            read_target = torch.zeros_like(source)
            read_memory = first.register_memory(read_target)
            self.assertEqual(
                first.read(
                    read_memory.to_mutable_view(),
                    destination_memory.to_remote_buffer(),
                ),
                0,
            )
            self.assertEqual(read_target, source)
        finally:
            first.close()
            second.close()

    def test_transport_partial_view_and_registration_reuse(self) -> None:
        first, second = self.make_transport_pair()
        try:
            source = torch.arange(64, dtype=torch.uint8)
            destination = torch.zeros(16, dtype=torch.uint8)
            source_memory = first.register_memory(source)
            destination_memory = second.register_memory(destination)

            self.assertFalse(source_memory.reused_registration())
            self.assertTrue(first.register_memory(source).reused_registration())
            self.assertEqual(
                first.write(
                    source_memory.to_view(16, 16),
                    destination_memory.to_remote_buffer(),
                ),
                0,
            )
            self.assertEqual(destination, source[16:32])
            with self.assertRaises(ValueError):
                source_memory.to_view(source.nbytes + 1)
        finally:
            first.close()
            second.close()

    def test_transport_async_work(self) -> None:
        first, second = self.make_transport_pair()
        try:
            source = torch.arange(64, dtype=torch.uint8)
            destination = torch.zeros_like(source)
            read_target = torch.zeros_like(source)
            source_memory = first.register_memory(source)
            destination_memory = second.register_memory(destination)
            read_memory = first.register_memory(read_target)
            descriptor = destination_memory.to_remote_buffer()
            remote = type(descriptor).deserialize(descriptor.serialize())
            write = first.write(source_memory.to_view(), remote, async_op=True)
            write.wait()
            read = first.read(read_memory.to_mutable_view(), remote, async_op=True)
            self.assertIsInstance(write, Work)
            self.assertIsInstance(read, Work)
            self.assertTrue(read.wait())
            self.assertTrue(write.wait())
            self.assertTrue(write.is_completed())
            self.assertTrue(read.is_completed())
            self.assertEqual(read.get_future().wait(), [])
            self.assertEqual(destination, source)
            self.assertEqual(read_target, source)
        finally:
            first.close()
            second.close()

    def test_transport_asyncio_put_get(self) -> None:
        client, server = self.make_transport_pair()
        try:
            source = torch.arange(64, dtype=torch.uint8)
            stored = torch.zeros_like(source)
            destination = torch.zeros_like(source)
            source_memory = client.register_memory(source)
            stored_memory = server.register_memory(stored)

            async def run():
                await server.read_async(
                    stored_memory.to_mutable_view(), source_memory.to_remote_buffer()
                )
                destination_memory = client.register_memory(destination)
                await server.write_async(
                    stored_memory.to_view(), destination_memory.to_remote_buffer()
                )
                await wait_all(
                    server.write(
                        stored_memory.to_view(),
                        destination_memory.to_remote_buffer(),
                        async_op=True,
                    )
                    for _ in range(2)
                )

            asyncio.run(run())
            self.assertEqual(stored, source)
            self.assertEqual(destination, source)
        finally:
            server.close()
            client.close()
