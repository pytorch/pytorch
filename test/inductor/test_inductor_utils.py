# Owner(s): ["module: inductor"]

import functools
import logging
import re

import torch
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import do_bench_using_profiling


log = logging.getLogger(__name__)


class TestBench(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        x = torch.rand(1024, 10).cuda().half()
        w = torch.rand(512, 10).cuda().half()
        cls._bench_fn = functools.partial(torch.nn.functional.linear, x, w)

    def test_benchmarker(self):
        res = benchmarker.benchmark_gpu(self._bench_fn)
        log.warning("do_bench result: %s", res)
        self.assertGreater(res, 0)

    def test_do_bench_using_profiling(self):
        res = do_bench_using_profiling(self._bench_fn)
        log.warning("do_bench_using_profiling result: %s", res)
        self.assertGreater(res, 0)


class TestPrecompiledPatterns(TestCase):
    """Tests for pre-compiled regex patterns in torch._inductor.utils."""

    def test_fused_abs_max_pattern_exists_and_precompiled(self):
        """Verify _FUSED_ABS_MAX_PATTERN is pre-compiled at module level."""
        from torch._inductor import utils

        self.assertTrue(
            hasattr(utils, "_FUSED_ABS_MAX_PATTERN"),
            "_FUSED_ABS_MAX_PATTERN should be defined at module level",
        )
        self.assertIsInstance(
            utils._FUSED_ABS_MAX_PATTERN,
            re.Pattern,
            "_FUSED_ABS_MAX_PATTERN should be a compiled regex pattern",
        )

    def test_fused_abs_max_pattern_matches_correctly(self):
        """Verify _FUSED_ABS_MAX_PATTERN matches expected strings."""
        from torch._inductor.utils import _FUSED_ABS_MAX_PATTERN

        # Should match
        self.assertIsNotNone(_FUSED_ABS_MAX_PATTERN.match("fused_abs_max_0"))
        self.assertIsNotNone(_FUSED_ABS_MAX_PATTERN.match("fused_abs_max_1"))
        self.assertIsNotNone(_FUSED_ABS_MAX_PATTERN.match("fused_abs_max_9"))
        self.assertIsNotNone(_FUSED_ABS_MAX_PATTERN.match("fused_abs_max_123"))

        # Should not match
        self.assertIsNone(_FUSED_ABS_MAX_PATTERN.match("fused_abs_max_"))
        self.assertIsNone(_FUSED_ABS_MAX_PATTERN.match("other_kernel"))
        self.assertIsNone(_FUSED_ABS_MAX_PATTERN.match(""))


if __name__ == "__main__":
    run_tests("cuda")
