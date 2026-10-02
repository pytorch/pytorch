# Owner(s): ["module: dsl-native-ops"]
# Host-side hardware-capability arithmetic and per-device caching.

import sys
import unittest

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_CUTEDSL, TestCase


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

from torch._native.cutedsl.hw_caps import caps


@unittest.skipUnless(TEST_CUDA, "CUDA required")
class TestHWCaps(TestCase):
    def _caps(self):
        return caps()

    def test_max_warps_per_sm(self):
        c = self._caps()
        self.assertEqual(c.max_warps_per_sm, c.max_threads_per_sm // c.warp)

    def test_peak_bw_positive(self):
        # bus_width/8 * memclock * 1e3 * 2 (DDR): a positive byte/s rate.
        self.assertGreater(self._caps().peak_bw_bytes, 0)

    def test_waves_scales_with_grid(self):
        # waves is linear: zero blocks is zero waves; doubling blocks doubles waves.
        c = self._caps()
        self.assertEqual(c.waves(0, 256), 0.0)
        self.assertEqual(c.waves(2000, 256), 2 * c.waves(1000, 256))

    def test_fill_blocks_is_one_wave(self):
        # fill_blocks(tpb, 1) is the concurrent-block count; N waves scales it by N.
        c = self._caps()
        for tpb in (128, 256, 1024):
            fill = c.fill_blocks(tpb, 1.0)
            self.assertEqual(fill, c.fill_blocks(tpb, 2.0) // 2)
            self.assertAlmostEqual(c.waves(fill, tpb), 1.0, places=5)

    def test_blocks_per_sm_floor(self):
        # The one-block floor keeps oversized blocks from making fill_blocks return zero.
        c = self._caps()
        huge = c.max_threads_per_sm * 4
        self.assertGreaterEqual(c.fill_blocks(huge, 1.0), c.sm_count)

    def test_caps_cached_per_device(self):
        self.assertIs(self._caps(), self._caps())


if __name__ == "__main__":
    run_tests()
