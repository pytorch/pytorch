# Owner(s): ["module: inductor"]

import functools
import logging
import time

import torch
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    do_bench_using_profiling,
    get_device_properties,
    get_max_num_sms,
    get_max_numwarps,
)


log = logging.getLogger(__name__)


class TestDevicePropertiesCache(TestCase):
    def test_get_device_properties_caching(self):
        get_device_properties.cache_clear()
        props1 = get_device_properties()
        props2 = get_device_properties()
        self.assertIs(props1, props2)
        self.assertTrue(hasattr(props1, "multi_processor_count"))
        self.assertGreater(props1.multi_processor_count, 0)

    def test_get_device_properties_repeated_calls_fast(self):
        get_device_properties.cache_clear()
        start = time.perf_counter()
        for _ in range(100):
            get_device_properties()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.01)

    def test_get_device_properties_different_devices(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 CUDA devices for this test")

        get_device_properties.cache_clear()
        props0 = get_device_properties(0)
        get_device_properties(1)
        self.assertIs(props0, get_device_properties(0))

    def test_get_max_num_sms_uses_cached_properties(self):
        get_device_properties.cache_clear()
        get_max_num_sms.cache_clear()
        self.assertGreater(get_max_num_sms(), 0)
        self.assertGreater(
            get_device_properties.cache_info().hits
            + get_device_properties.cache_info().misses,
            0,
        )

    def test_get_max_numwarps_uses_cached_properties(self):
        get_device_properties.cache_clear()
        get_max_numwarps.cache_clear()
        warps = get_max_numwarps()
        self.assertGreater(warps, 0)
        self.assertGreaterEqual(warps, 1)
        self.assertLessEqual(warps, 64)


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


if __name__ == "__main__":
    run_tests("cuda")
