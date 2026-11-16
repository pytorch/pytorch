import asyncio
import threading
from datetime import timedelta

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

            self.assertEqual(
                source_memory.to_remote_buffer(),
                first.register_memory(source).to_remote_buffer(),
            )
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

    def test_transport_slice(self) -> None:
        first, second = self.make_transport_pair()
        try:
            source = torch.arange(64, dtype=torch.uint8)
            destination = torch.zeros_like(source)
            read_target = torch.zeros(16, dtype=torch.uint8)
            source_memory = first.register_memory(source)
            read_memory = first.register_memory(read_target)
            descriptor = second.register_memory(destination).to_remote_buffer()
            remote = type(descriptor).deserialize(descriptor.serialize())
            part = remote.slice(16, 16)
            part = type(part).deserialize(part.serialize())
            self.assertEqual(part.serialize(), remote[16:32].serialize())
            self.assertEqual(part.serialize(), remote[-48:-32].serialize())

            first.write(source_memory.to_view()[8:24], part)
            self.assertEqual(destination[16:32], source[8:24])
            self.assertEqual(destination[:16].count_nonzero(), 0)
            self.assertEqual(destination[32:].count_nonzero(), 0)
            view = read_memory.to_mutable_view()[8:]
            self.assertIsInstance(view, MutableMemoryView)
            self.assertNotIsInstance(source_memory.to_view()[8:], MutableMemoryView)
            first.read(view.slice(0, 4), part.slice(4)[:4])
            self.assertEqual(read_target[8:12], source[12:16])
            self.assertEqual(read_target[:8].count_nonzero(), 0)
            self.assertEqual(read_target[12:].count_nonzero(), 0)
            self.assertEqual(part[4:100].serialize(), part.slice(4).serialize())
            self.assertEqual(source_memory.to_view()[60:100].size(), 4)
            self.assertEqual(source_memory.to_view()[40:20].size(), 0)
            with self.assertRaisesRegex(ValueError, "does not fit"):
                first.write(source_memory.to_view(0, 17), part)
            for sliceable, size in ((part, 16), (source_memory.to_view(0, 16), 16)):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    sliceable.slice(8, 9)
                for offset in (size + 1, -1):
                    with self.assertRaisesRegex(ValueError, "outside"):
                        sliceable.slice(offset)
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    sliceable.slice(0, -1)
                with self.assertRaisesRegex(ValueError, "step"):
                    sliceable[::2]
                with self.assertRaisesRegex(TypeError, "expected a slice"):
                    sliceable[0]
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

    def _notification_pair(self):
        first, second = self.make_transport_pair()
        if not first.max_notification_size():
            first.close()
            second.close()
            self.skipTest("transport does not support notifications")
        return first, second

    def _receive(self, transport, count=1):
        received = []
        while len(received) < count:
            transport.wait_notifications().wait(timedelta(seconds=30))
            received += transport.poll_notifications()
        return received

    def test_notification_round_trip(self) -> None:
        first, second = self._notification_pair()
        try:
            payload = b"\x00\xff" + bytes(range(256))
            self.assertTrue(first.send_notification(second.peer_id, payload).wait())
            received = self._receive(second)
            self.assertEqual(received, [(first.peer_id, payload)])
            self.assertEqual(second.poll_notifications(), [])
            second.send_notification(received[0].peer, b"ack").wait()
            self.assertEqual(self._receive(first), [(second.peer_id, b"ack")])
        finally:
            first.close()
            second.close()

    def test_notification_validation(self) -> None:
        first, second = self._notification_pair()
        try:
            limit = first.max_notification_size()
            for payload in ("text", bytearray(b"x")):
                with self.assertRaisesRegex(TypeError, "must be bytes"):
                    first.send_notification(second.peer_id, payload)
            for payload in (b"", bytes(limit + 1)):
                with self.assertRaisesRegex(ValueError, "payload must be"):
                    first.send_notification(second.peer_id, payload)
            with self.assertRaisesRegex(ValueError, "not connected"):
                first.send_notification("unknown-peer", b"x")
            first.send_notification(second.peer_id, bytes(limit)).wait()
            self.assertEqual(self._receive(second), [(first.peer_id, bytes(limit))])
        finally:
            first.close()
            second.close()

    def test_transfer_notifications(self) -> None:
        first, second = self._notification_pair()
        try:
            source = torch.arange(64, dtype=torch.uint8)
            destination = torch.zeros_like(source)
            read_target = torch.zeros_like(source)
            view = first.register_memory(source).to_view()
            read_view = first.register_memory(read_target).to_mutable_view()
            remote = second.register_memory(destination).to_remote_buffer()
            self.assertEqual(remote.peer_id, second.peer_id)

            self.assertEqual(first.write(view, remote, notification=b"write"), 0)
            self.assertEqual(self._receive(second), [(first.peer_id, b"write")])
            self.assertEqual(destination, source)
            self.assertEqual(first.read(read_view, remote, notification=b"read"), 0)
            self.assertEqual(self._receive(second), [(first.peer_id, b"read")])
            self.assertEqual(read_target, source)

            destination.zero_()
            work = first.write(view, remote, async_op=True, notification=b"work")
            work.wait()
            self.assertEqual(self._receive(second), [(first.peer_id, b"work")])
            self.assertEqual(destination, source)
            destination.zero_()
            asyncio.run(first.write_async(view, remote, notification=b"coroutine"))
            self.assertEqual(self._receive(second), [(first.peer_id, b"coroutine")])
            self.assertEqual(destination, source)

            with self.assertRaisesRegex(ValueError, "zero-byte"):
                first.write(view[:0], remote, notification=b"x")
            first.send_notification(second.peer_id, b"after").wait()
            self.assertEqual(self._receive(second), [(first.peer_id, b"after")])
            self.assertEqual(second.poll_notifications(), [])
        finally:
            first.close()
            second.close()

    def test_wait_notifications(self) -> None:
        first, second = self._notification_pair()
        try:
            work = second.wait_notifications()
            self.assertFalse(work.is_completed())
            with self.assertRaises(TimeoutError):
                work.wait(timedelta(milliseconds=10))
            first.send_notification(second.peer_id, b"late").wait()
            self.assertTrue(work.wait(timedelta(seconds=30)))
            self.assertEqual(second.poll_notifications(), [(first.peer_id, b"late")])

            first.send_notification(second.peer_id, b"asyncio").wait()
            asyncio.run(wait_all([second.wait_notifications()], timeout=30))
            self.assertEqual(self._receive(second), [(first.peer_id, b"asyncio")])

            chained = second.wait_notifications().get_future().then(lambda _: None)
            done = threading.Event()
            chained.add_done_callback(lambda _: done.set())
            first.send_notification(second.peer_id, b"chained").wait()
            self.assertTrue(done.wait(30))
            self.assertEqual(self._receive(second), [(first.peer_id, b"chained")])

            pending = second.wait_notifications()
            second.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pending.wait()
        finally:
            first.close()
            second.close()

    def test_notification_burst(self) -> None:
        first, second = self._notification_pair()
        try:
            count = 500
            received = []
            receiver = threading.Thread(
                target=lambda: received.extend(self._receive(second, count))
            )
            receiver.start()
            works = [
                first.send_notification(second.peer_id, str(i).encode())
                for i in range(count)
            ]
            for work in works:
                work.wait()
            receiver.join()
            self.assertEqual(
                sorted(int(payload) for _, payload in received), list(range(count))
            )
            self.assertEqual({peer for peer, _ in received}, {first.peer_id})
        finally:
            first.close()
            second.close()
