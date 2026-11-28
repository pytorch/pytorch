# Owner(s): ["module: tests"]

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcContinuousTest
from torch.testing._internal.common_utils import run_tests, TestCase


class TestResolveInjectedDeviceUnit(TestCase):
    """Unit test for ``MultiProcContinuousTest._resolve_injected_device``: the
    framework method that rewrites the rank-0 injected device to this worker's
    per-rank device (called by ``instantiate_device_type_tests``' wrapper)."""

    @staticmethod
    def _make(rank):
        inst = MultiProcContinuousTest.__new__(MultiProcContinuousTest)
        inst.rank = rank
        return inst

    def test_accelerator_device_rebinds_to_rank(self):
        # The injected "{type}:0" is rebound to the string "{type}:{rank}" (the
        # injected `device` arg is a str, so the param keeps its type). "cuda:0"
        # is portable: the cuda device type is always recognized and
        # torch.device("cuda:0") is lazy (no CUDA hardware required).
        self.assertEqual(self._make(3)._resolve_injected_device("cuda:0"), "cuda:3")

    def test_cpu_passes_through(self):
        # CPU is a single shared device with no per-rank split; pass through.
        self.assertEqual(self._make(3)._resolve_injected_device("cpu"), "cpu")


class TestRankAwareDeviceInjectionHook(TestCase):
    """Self-test for the call-time hook in ``instantiate_device_type_tests``'
    wrapper: when a test class defines ``_resolve_injected_device``, the injected
    ``device`` arg is rewritten to carry ``self.rank`` before the test body runs.

    Single-process (a plain ``TestCase``, not ``MultiProcContinuousTest``) so it
    exercises the wrapper hook without the multi-process spawn path. The
    ``device`` is compared as a string (no tensor placement), so the rank-N
    device need not physically exist -- runs on a single accelerator, no
    multi-accelerator requirement.
    """

    rank = 1  # simulate a non-zero multi-process rank

    def _resolve_injected_device(self, device):
        dev = torch.device(device)
        if dev.type == "cpu":
            return device
        return f"{dev.type}:{self.rank}"

    def test_injected_device_carries_rank(self, device):
        # The wrapper rewrote the injected "{type}:0" to "{type}:{rank}".
        self.assertEqual(device, f"{torch.device(device).type}:{self.rank}")


instantiate_device_type_tests(
    TestRankAwareDeviceInjectionHook, globals(), except_for=["cpu"]
)


if __name__ == "__main__":
    run_tests()
