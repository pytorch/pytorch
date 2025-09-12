# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class TestEvent(TestCase):
    @skipIfTorchDynamo()
    def test_record_event(self):
        stream = torch.Stream(device="openreg:1")
        event1 = stream.record_event()
        self.assertNotEqual(0, event1.event_id)
        event2 = stream.record_event()
        self.assertNotEqual(0, event2.event_id)
        self.assertNotEqual(event1.event_id, event2.event_id)

    @skipIfTorchDynamo()
    def test_event_elapsed_time(self):
        stream = torch.Stream(device="openreg:1")
        e1 = torch.Event(device="openreg:1", enable_timing=True)
        e1.record(stream)
        e2 = torch.Event(device="openreg:1", enable_timing=True)
        e2.record(stream)

        e2.synchronize()
        self.assertTrue(e2.query())

        ms = e1.elapsed_time(e2)
        self.assertTrue(ms > 0)

    @skipIfTorchDynamo()
    def test_event_wait_stream(self):
        s1 = torch.Stream(device="openreg")
        s2 = torch.Stream(device="openreg")
        e1 = s1.record_event()
        e1.wait(s2)


if __name__ == "__main__":
    run_tests()
