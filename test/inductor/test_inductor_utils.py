# Owner(s): ["module: inductor"]

import functools
import logging

import torch
from torch._inductor import config
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import do_bench_using_profiling
from torch.testing._internal.logging_utils import logs_to_string


try:
    from .test_codecache import capture_logs
except ImportError:
    from test_codecache import capture_logs


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


@config.patch("run_with_post_grad_graph", True)
class TestPostGradRun(TestCase):
    def test_post_grad_run(self):
        post_grad_log_stream, post_grad_log_ctx = logs_to_string(
            "torch._inductor.compile_fx", "post_grad_graphs"
        )

        class PostGrad(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                return self.linear(x)

        model = PostGrad()
        inp = (torch.randn(10, 10),)
        res1 = model(*inp)

        with (
            capture_logs("torch._inductor.scheduler", logging.DEBUG) as scheduler_logs,
            post_grad_log_ctx(),
        ):
            res2 = torch.compile(model)(*inp)

        self.assertTrue(torch.allclose(res1, res2))

        post_grad_log = "\n".join(
            post_grad_log_stream.getvalue().strip().split("\n")[3:]
        ).strip()
        self.assertEqual(scheduler_logs, [])
        self.assertTrue(len(post_grad_log) > 0)


if __name__ == "__main__":
    run_tests("cuda")
