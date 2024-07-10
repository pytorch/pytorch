# Owner(s): ["module: inductor"]

import functools
import logging

import torch
from torch._inductor.runtime.runtime_utils import do_bench

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

    def test_do_bench(self):
        res = do_bench(self._bench_fn)
        log.warning("do_bench result: %s", res)
        self.assertGreater(res, 0)

    def test_do_bench_using_profiling(self):
        res = do_bench_using_profiling(self._bench_fn)
        log.warning("do_bench_using_profiling result: %s", res)
        self.assertGreater(res, 0)


if __name__ == "__main__":
    run_tests("cuda")
