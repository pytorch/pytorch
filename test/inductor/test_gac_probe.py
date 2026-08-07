# Owner(s): ["module: ci"]
"""Control probe for the GPU_ARCH_COVERAGE linter. DO NOT LAND."""

from torch.testing._internal.common_utils import run_tests, TestCase


class GacProbeControlTest(TestCase):
    def test_gac_probe_no_arch_gate(self):
        self.assertTrue(True)


if __name__ == "__main__":
    run_tests()
