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


if __name__ == "__main__":
    run_tests()
