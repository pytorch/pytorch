# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestEvent(TestCase):
    @skipIfTorchDynamo()
    def test_event_create(self):
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
        event = torch.Event()
        self.assertTrue(event.query())

        stream = torch.Stream(device="openreg:1")
        event = stream.record_event()
        event.synchronize()
        self.assertTrue(event.query())

    @skipIfTorchDynamo()
    def test_event_record(self):
        stream = torch.Stream(device="openreg:1")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)

        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)

        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfTorchDynamo()
    def test_event_elapsed_time(self):
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
        stream1 = torch.Stream(device="openreg")
        stream2 = torch.Stream(device="openreg")

        event = stream1.record_event()
        stream2.wait_event(event)


if __name__ == "__main__":
    run_tests()
