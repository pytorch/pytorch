# Owner(s): ["module: dsl-native-ops"]
#
# Unit tests for the pure-arithmetic surface of _cutedsl.hw_caps.HWCaps -- the
# derived occupancy quantities the launch heuristics reason in. These are integer
# formulas over device properties (no kernel launch), so they are checked here for
# internal consistency and against hand-computed values; the rest of the _cutedsl
# machinery (traits / launch) only does work once compiled into a kernel and is
# exercised by the reduction-override suites that build on it.

import unittest

from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestHWCaps(TestCase):
    def _caps(self):
        from torch._native.ops._cutedsl.hw_caps import caps

        return caps()

    def test_max_warps_per_sm(self):
        c = self._caps()
        self.assertEqual(c.max_warps_per_sm, c.max_threads_per_sm // c.warp)

    def test_peak_bw_positive(self):
        # bus_width/8 * memclock * 1e3 * 2 (DDR): a positive byte/s rate.
        self.assertGreater(self._caps().peak_bw_bytes, 0)

    def test_waves_scales_with_grid(self):
        # waves is linear in total_blocks: an empty grid is 0 waves, and doubling
        # the block count doubles the wave count for a fixed block size.
        c = self._caps()
        self.assertEqual(c.waves(0, 256), 0.0)
        self.assertEqual(c.waves(2000, 256), 2 * c.waves(1000, 256))

    def test_fill_blocks_is_one_wave(self):
        # fill_blocks(tpb, 1.0) is exactly the concurrent-block count, i.e. the grid
        # whose waves() is 1.0; and N waves scales it by N.
        c = self._caps()
        for tpb in (128, 256, 1024):
            fill = c.fill_blocks(tpb, 1.0)
            self.assertEqual(fill, c.fill_blocks(tpb, 2.0) // 2)
            self.assertAlmostEqual(c.waves(fill, tpb), 1.0, places=5)

    def test_blocks_per_sm_floor(self):
        # A block larger than the SM thread cap still yields >= 1 block/SM (the
        # max(1, ...) floor), so fill_blocks never collapses to 0.
        c = self._caps()
        huge = c.max_threads_per_sm * 4
        self.assertGreaterEqual(c.fill_blocks(huge, 1.0), c.sm_count)

    def test_caps_cached_per_device(self):
        # caps() memoizes per device index -> same object back.
        self.assertIs(self._caps(), self._caps())


if __name__ == "__main__":
    run_tests()
