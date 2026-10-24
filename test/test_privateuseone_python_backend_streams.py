# Owner(s): ["module: PrivateUse1"]

import torch
import torch._C
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend


_DEVICE_TYPE = torch._C._autograd.DeviceType.PrivateUse1


class StreamDeviceGuard(torch._C._acc.DeviceGuard):
    """A python-only backend that defines how to create a stream."""

    def __init__(self):
        super().__init__()
        self.next_id = 0
        self.calls = []

    def type_(self):
        return _DEVICE_TYPE

    def get_new_stream(self, device, priority):
        self.calls.append((device.type, device.index, priority))
        self.next_id += 1
        # NOTE: the stream has to be built with the
        # (stream_id, device_index, device_type) overload. Stream(device=...)
        # would call back into this method and recurse forever.
        index = device.index if device.index is not None else 0
        return torch.Stream(self.next_id, index, int(_DEVICE_TYPE))


# NOTE: device guard registration is process-global and first-wins, so this
# guard must be registered before _setup_privateuseone_for_python_backend
# installs its default one, and only one guard can be tested per process.
_GUARD = StreamDeviceGuard()
torch._C._acc.register_python_privateuseone_device_guard(_GUARD)
_setup_privateuseone_for_python_backend("mydev")


@skipIfTorchDynamo("stream creation is device plumbing, not traceable work")
class PrivateUse1PythonStreamTest(TestCase):
    def test_new_stream_dispatches_to_python(self):
        before = len(_GUARD.calls)
        stream = torch.Stream(device="mydev")
        self.assertEqual(len(_GUARD.calls), before + 1)
        # The old no-op handed back the default stream, whose id is 0.
        self.assertNotEqual(stream.stream_id, 0)

    def test_new_streams_are_distinct(self):
        s1 = torch.Stream(device="mydev")
        s2 = torch.Stream(device="mydev")
        self.assertNotEqual(s1.stream_id, s2.stream_id)

    def test_priority_is_forwarded(self):
        torch.Stream(device="mydev", priority=-1)
        self.assertIn(-1, [priority for _, _, priority in _GUARD.calls])

    def test_stream_keeps_device(self):
        s = torch.Stream(device="mydev")
        self.assertEqual(s.device.type, "mydev")


if __name__ == "__main__":
    run_tests()
