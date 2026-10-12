# Owner(s): ["module: PrivateUse1"]

import multiprocessing
import unittest

import torch
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)


class TestStream(TestCase):
    @skipIfTorchDynamo()
    def test_stream_create(self):
        """Test stream creation with different methods"""
        stream = torch.Stream(device="openreg")
        self.assertEqual(stream.device_index, torch.openreg.current_device())
        stream = torch.Stream(device="openreg:1")
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream = torch.Stream(1)
        self.assertEqual(stream.device.type, "openreg")
        self.assertEqual(stream.device_index, 1)

        stream1 = torch.Stream(
            stream_id=stream.stream_id,
            device_type=stream.device_type,
            device_index=stream.device_index,
        )
        self.assertEqual(stream, stream1)

    @skipIfTorchDynamo()
    def test_stream_context(self):
        """Test stream context manager"""
        with torch.Stream(device="openreg:1") as stream:
            self.assertEqual(torch.accelerator.current_stream(), stream)

    def test_stream_context_exception_restore(self):
        prev = torch.accelerator.current_stream()
        inner_stream = torch.Stream(device="openreg:1")
        try:
            with inner_stream:
                # inside the context we should be on the inner stream
                self.assertEqual(torch.accelerator.current_stream(), inner_stream)
                raise RuntimeError("forced")
        except RuntimeError:
            pass
        # After the exception, the current stream should be restored.
        self.assertEqual(torch.accelerator.current_stream(), prev)

    @skipIfTorchDynamo()
    def test_stream_switch(self):
        """Test switching between streams"""
        stream1 = torch.Stream(device="openreg:0")
        torch.accelerator.set_stream(stream1)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)

        stream2 = torch.Stream(device="openreg:1")
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream1)
        torch.accelerator.set_stream(stream2)
        current_stream = torch.accelerator.current_stream()
        self.assertEqual(current_stream, stream2)

    @skipIfTorchDynamo()
    def test_stream_synchronize(self):
        """Test stream synchronization"""
        stream = torch.Stream(device="openreg:1")
        self.assertEqual(True, stream.query())

        event = torch.Event()
        event.record(stream)
        stream.synchronize()
        self.assertEqual(True, stream.query())

    @skipIfTorchDynamo()
    def test_stream_repr(self):
        """Test stream string representation"""
        stream = torch.Stream(device="openreg:1")
        self.assertTrue(
            "torch.Stream device_type=openreg, device_index=1" in repr(stream)
        )

    @skipIfTorchDynamo()
    def test_stream_wait_stream(self):
        """Test stream waiting on another stream"""
        stream_1 = torch.Stream(device="openreg:0")
        stream_2 = torch.Stream(device="openreg:1")
        stream_2.wait_stream(stream_1)

    @skipIfTorchDynamo()
    def test_stream_wait_event(self):
        """Test stream waiting on event"""
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e = s1.record_event()
        s2.wait_event(e)

    @skipIfTorchDynamo()
    def test_stream_equality(self):
        """Test stream equality comparison"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")

        # Different streams should not be equal
        self.assertNotEqual(stream1, stream2)

        # Same stream should be equal to itself
        self.assertEqual(stream1, stream1)

        # Stream created with same parameters should be equal
        stream3 = torch.Stream(
            stream_id=stream1.stream_id,
            device_type=stream1.device_type,
            device_index=stream1.device_index,
        )
        self.assertEqual(stream1, stream3)

    @skipIfTorchDynamo()
    def test_stream_multiple_devices(self):
        """Test streams on multiple devices"""
        stream0 = torch.Stream(device="openreg:0")
        stream1 = torch.Stream(device="openreg:1")

        self.assertEqual(stream0.device_index, 0)
        self.assertEqual(stream1.device_index, 1)

        # Set current stream for each device
        torch.accelerator.set_device_index(0)
        torch.accelerator.set_stream(stream0)
        self.assertEqual(torch.accelerator.current_stream(), stream0)

        torch.accelerator.set_device_index(1)
        torch.accelerator.set_stream(stream1)
        self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_context_nested(self):
        """Test nested stream contexts"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")

        with stream1:
            self.assertEqual(torch.accelerator.current_stream(), stream1)
            with stream2:
                self.assertEqual(torch.accelerator.current_stream(), stream2)
            # Should restore to stream1
            self.assertEqual(torch.accelerator.current_stream(), stream1)

    @skipIfTorchDynamo()
    def test_stream_record_event(self):
        """Test recording events on streams"""
        stream = torch.Stream(device="openreg")
        event = stream.record_event()

        self.assertIsNotNone(event)
        self.assertEqual(event.device.type, "openreg")
        stream.synchronize()
        self.assertTrue(event.query())


@unittest.skipIf(IS_WINDOWS, "OpenReg IPC events require POSIX shared memory")
class TestOpenRegMultiprocessing(TestCase):
    @staticmethod
    def _event_ipc_importer_consumer(handle, p2c, c2p):
        e1 = torch.Event.from_ipc_handle(
            torch.accelerator.current_device_index(), handle
        )
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for parent to record the event
        e1.synchronize()
        c2p.put(0)  # notify parent synchronization is done

    @staticmethod
    def _event_ipc_exporter_consumer(handle, p2c, c2p):
        stream = torch.Stream(device="openreg")
        e1 = torch.Event.from_ipc_handle(
            torch.accelerator.current_device_index(), handle
        )
        e1.record(stream)
        c2p.put(0)  # notify parent the event has been recorded
        p2c.get()  # wait for parent to finish

    def test_event_ipc_importer(self):
        """Parent records an IPC event; a child process imports the handle
        and synchronizes on it through the shared event state."""
        e0 = torch.Event(enable_timing=False, interprocess=True)

        ctx = multiprocessing.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestOpenRegMultiprocessing._event_ipc_importer_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for child to become ready
        stream = torch.Stream(device="openreg")
        e0.record(stream)
        p2c.put(0)  # notify child the event has been recorded

        c2p.get()  # wait for synchronization in child
        p.join()

    def test_event_ipc_exporter(self):
        """A child process imports an IPC event and records it; the parent
        (exporter) synchronizes on it through the shared event state."""
        e0 = torch.Event(enable_timing=False, interprocess=True)

        ctx = multiprocessing.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestOpenRegMultiprocessing._event_ipc_exporter_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for child to record the event
        e0.synchronize()
        p2c.put(0)  # notify child to finish
        p.join()


if __name__ == "__main__":
    run_tests()
