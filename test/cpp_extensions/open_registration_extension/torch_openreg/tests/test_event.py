# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestEvent(TestCase):
    @skipIfTorchDynamo()
    def test_event_create(self):
        """Test event creation with different methods"""
        event = torch.Event(device="openreg")
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = torch.Event(device="openreg:1")
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        event = torch.Event()
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, None)
        self.assertEqual(event.event_id, 0)

        stream = torch.Stream(device="openreg:1")
        event = stream.record_event()
        self.assertEqual(event.device.type, "openreg")
        self.assertEqual(event.device.index, 1)
        self.assertNotEqual(event.event_id, 0)

    @skipIfTorchDynamo()
    def test_event_query(self):
        """Test event query operation"""
        event = torch.Event()
        self.assertTrue(event.query())

        stream = torch.Stream(device="openreg:1")
        event = stream.record_event()
        event.synchronize()
        self.assertTrue(event.query())

    @skipIfTorchDynamo()
    def test_event_record(self):
        """Test recording events on streams"""
        stream = torch.Stream(device="openreg:1")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)

        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)

        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfTorchDynamo()
    def test_event_elapsed_time(self):
        """Test elapsed time calculation between events"""
        stream = torch.Stream(device="openreg:1")

        event1 = torch.Event(device="openreg:1", enable_timing=True)
        event1.record(stream)
        event2 = torch.Event(device="openreg:1", enable_timing=True)
        event2.record(stream)

        stream.synchronize()
        self.assertTrue(event1.query())
        self.assertTrue(event2.query())

        ms = event1.elapsed_time(event2)
        self.assertTrue(ms > 0)

    @skipIfTorchDynamo()
    def test_event_wait_stream(self):
        """Test stream waiting on event"""
        stream1 = torch.Stream(device="openreg")
        stream2 = torch.Stream(device="openreg")

        event = stream1.record_event()
        stream2.wait_event(event)

    @skipIfTorchDynamo()
    def test_event_synchronize(self):
        """Test event synchronization"""
        event = torch.Event(device="openreg")
        self.assertTrue(event.query())

        stream = torch.Stream(device="openreg")
        event.record(stream)
        event.synchronize()
        self.assertTrue(event.query())

    @skipIfTorchDynamo()
    def test_event_different_devices(self):
        """Test events on different devices"""
        event0 = torch.Event(device="openreg:0")
        event1 = torch.Event(device="openreg:1")

        stream0 = torch.Stream(device="openreg:0")
        stream1 = torch.Stream(device="openreg:1")

        event0.record(stream0)
        event1.record(stream1)

        self.assertEqual(event0.device.index, 0)
        self.assertEqual(event1.device.index, 1)

    @skipIfTorchDynamo()
    def test_event_timing_disabled(self):
        """Test event with timing disabled"""
        event1 = torch.Event(device="openreg:1", enable_timing=False)
        event2 = torch.Event(device="openreg:1", enable_timing=False)

        stream = torch.Stream(device="openreg:1")
        event1.record(stream)
        event2.record(stream)
        stream.synchronize()

        # Should not be able to calculate elapsed time
        with self.assertRaisesRegex(
            ValueError,
            "Both events must be created with argument 'enable_timing=True'.",
        ):
            _ = event1.elapsed_time(event2)

    @skipIfTorchDynamo()
    def test_event_wait_event(self):
        """Test stream waiting on event"""
        stream1 = torch.Stream(device="openreg")
        stream2 = torch.Stream(device="openreg")

        event = stream1.record_event()
        stream2.wait_event(event)
        stream2.synchronize()

        self.assertTrue(event.query())

    @skipIfTorchDynamo()
    def test_device_synchronization(self):
        """Test device synchronization operations"""
        torch.accelerator.set_device_index(0)

        # Perform operations
        x = torch.randn(100, 100, device="openreg")
        y = torch.randn(100, 100, device="openreg")
        z = torch.matmul(x, y)

        # Synchronize device
        torch.accelerator.synchronize()
        torch.accelerator.synchronize(0)  # Explicit device index

        # Verify operations completed
        result = torch.sum(z)
        self.assertIsNotNone(result)

    @skipIfTorchDynamo()
    def test_blocking_vs_non_blocking_synchronization(self):
        """Test blocking vs non-blocking event synchronization"""
        stream = torch.Stream(device="openreg:0")

        # test non-blocking query before recording
        event = torch.Event(device="openreg:0")
        self.assertTrue(event.query())  # Non-blocking, returns immediately

        # test non-blocking query after recording but before completion
        event.record(stream)
        # query() is non-blocking, may return False if event not completed
        _ = event.query()  # Non-blocking check

        # test blocking synchronize - waits for completion
        event.synchronize()  # Blocking - CPU waits here
        self.assertTrue(event.query())  # Should be True after sync

        # test multiple non-blocking queries
        event2 = stream.record_event()
        for _ in range(5):
            _ = event2.query()  # Non-blocking, doesn't wait
        event2.synchronize()  # Blocking wait
        self.assertTrue(event2.query())

    @skipIfTorchDynamo()
    def test_interleaved_stream_operations(self):
        """Test events with interleaved stream operations"""
        stream = torch.Stream(device="openreg:0")

        # create tensors for operations
        x = torch.randn(50, 50, device="openreg:0")
        y = torch.randn(50, 50, device="openreg:0")

        # record event before operations
        event1 = stream.record_event()

        # perform operations between events
        z1 = torch.matmul(x, y)

        # record second event
        event2 = stream.record_event()

        # more operations
        z2 = torch.matmul(z1, x)

        # record third event
        event3 = stream.record_event()

        # synchronize stream to ensure all operations complete
        stream.synchronize()

        # verify all events completed
        self.assertTrue(event1.query())
        self.assertTrue(event2.query())
        self.assertTrue(event3.query())

        # verify results are valid
        self.assertIsNotNone(torch.sum(z2))

    @skipIfTorchDynamo()
    def test_multiple_streams_wait_same_event(self):
        """Test multiple streams waiting on the same event"""
        stream1 = torch.Stream(device="openreg:0")
        stream2 = torch.Stream(device="openreg:0")
        stream3 = torch.Stream(device="openreg:0")

        # create tensors
        x = torch.randn(30, 30, device="openreg:0")
        y = torch.randn(30, 30, device="openreg:0")

        # record event on first stream
        event = stream1.record_event()

        # make multiple streams wait for the same event
        stream2.wait_event(event)
        stream3.wait_event(event)

        # perform operations on waiting streams
        with stream2:
            z2 = torch.matmul(x, y)

        with stream3:
            z3 = torch.matmul(x, y)

        # synchronize all streams
        stream1.synchronize()
        stream2.synchronize()
        stream3.synchronize()

        # verify event completed and results are valid
        self.assertTrue(event.query())
        self.assertIsNotNone(torch.sum(z2))
        self.assertIsNotNone(torch.sum(z3))

    @skipIfTorchDynamo()
    def test_event_lifecycle(self):
        """Test event creation, use, and destruction patterns"""
        # test 1: create unrecorded event
        event1 = torch.Event(device="openreg:0")
        self.assertEqual(event1.event_id, 0)
        self.assertTrue(event1.query())

        # test 2: record and use event
        stream = torch.Stream(device="openreg:0")
        event1.record(stream)
        self.assertNotEqual(event1.event_id, 0)
        event1.synchronize()
        self.assertTrue(event1.query())

        # test 3: reuse event after synchronization
        event1.record(stream)
        stream.synchronize()
        self.assertTrue(event1.query())

        # test 4: create multiple events in sequence
        events = []
        for _ in range(5):
            event = torch.Event(device="openreg:0")
            event.record(stream)
            events.append(event)

        stream.synchronize()

        # verify all events completed
        for event in events:
            self.assertTrue(event.query())

        # test 5: event with operations between creation and recording
        event2 = torch.Event(device="openreg:0")
        x = torch.randn(20, 20, device="openreg:0")
        y = torch.randn(20, 20, device="openreg:0")
        _ = torch.matmul(x, y)
        event2.record(stream)
        stream.synchronize()
        self.assertTrue(event2.query())

        # test 6: event recorded multiple times
        event3 = torch.Event(device="openreg:0")
        event3.record(stream)
        stream.synchronize()
        event3.record(stream)  # record again
        stream.synchronize()
        self.assertTrue(event3.query())


if __name__ == "__main__":
    run_tests()
