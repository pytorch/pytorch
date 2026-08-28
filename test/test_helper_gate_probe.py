# Owner(s): ["module: ci"]
from test_arch_gate_base import ArchGatedBase

from torch.testing._internal.common_utils import run_tests


class TestHelperGate(ArchGatedBase):
    def setUp(self):
        super().setUp()
        self._require_sm90()

    def test_gated_via_base_class_helper(self):
        self.assertTrue(True)


if __name__ == "__main__":
    run_tests()
