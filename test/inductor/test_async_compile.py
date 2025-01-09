# Owner(s): ["module: inductor"]
import torch
from torch._inductor import config
from torch._inductor.async_compile import AsyncCompile, shutdown_compile_workers
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import requires_gpu, requires_triton


@instantiate_parametrized_tests
class TestAsyncCompile(TestCase):
    @requires_gpu()
    @requires_triton()
    @parametrize("method", ("subprocess", "fork", "spawn"))
    def test_pool(self, method):
        def fn(x, y):
            return x + y

        x = torch.rand(10).cuda()
        y = torch.rand(10).cuda()

        with config.patch("worker_start_method", method):
            shutdown_compile_workers()
            pool = AsyncCompile.process_pool()
            pool.ready_future.result(timeout=120)

            with fresh_inductor_cache():
                compiled_fn = torch.compile(fn)
                self.assertEqual(fn(x, y), compiled_fn(x, y))


if __name__ == "__main__":
    run_tests()
