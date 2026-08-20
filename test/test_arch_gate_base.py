# Owner(s): ["module: ci"]
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import run_tests, TestCase


class ArchGatedBase(TestCase):
    """Shared base whose helper carries the gate, in a different module."""

    def _require_sm90(self):
        if not SM90OrLater:
            self.skipTest("needs sm90")


if __name__ == "__main__":
    run_tests()
