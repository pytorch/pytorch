# Owner(s): ["module: inductor"]
import contextlib
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    IS_BIG_GPU,
)


@instantiate_parametrized_tests
class DeterministicTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        self._exit_stack.close()
        super().tearDown()

    @parametrize("deterministic", [False, True])
    def test_mm_padding(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2049, 2049], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["pad_mm_bench"] > 0)

    @parametrize("deterministic", [False, True])
    @inductor_config.patch(max_autotune=True)
    @unittest.skipIf(not IS_BIG_GPU, "templates require big gpu")
    def test_max_autotune(self, deterministic):
        with inductor_config.patch(deterministic=deterministic):

            @torch.compile()
            def foo(x, y):
                return x @ y

            inps = [torch.rand([2048, 2048], device=GPU_TYPE) for _ in range(2)]
            out = foo(*inps)
            self.assertEqual(out, inps[0] @ inps[1])

            if deterministic:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] == 0)
            else:
                self.assertTrue(counters["inductor"]["select_algorithm_autotune"] > 0)

    def test_pointwise_coordesc_tuning(self):
        @torch.compile(mode="max-autotune")
        def f(x):
            return x + 1

        x = torch.randn(2048, device=GPU_TYPE)
        self.assertEqual(f(x), x + 1)

        self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)

    @parametrize("deterministic", [False, True])
    def test_reduction_coordesc_tuning(self, deterministic):
        with inductor_config.patch(
            deterministic=deterministic, coordinate_descent_tuning=True
        ):

            @torch.compile()
            def foo(x):
                return x.sum(dim=-1)

            inp = torch.rand([2048, 2048], device=GPU_TYPE)

            from torch._inductor.runtime.triton_heuristics import CachingAutotuner

            old_init = CachingAutotuner.__init__
            caching_autotuners = []

            def new_init(self, *args, **kwargs):
                old_init(self, *args, **kwargs)
                nonlocal caching_autotuners
                caching_autotuners.append(self)

            with unittest.mock.patch.object(CachingAutotuner, "__init__", new_init):
                out = foo(inp)

            self.assertEqual(out, inp.sum(dim=-1))

            self.assertEqual(len(caching_autotuners), 1)

            coordesc_tuner = caching_autotuners[0].coordesc_tuner
            frozen_fields = coordesc_tuner.frozen_fields

            if deterministic:
                self.assertTrue(len(frozen_fields) > 0)
                self.assertTrue("R0_BLOCK" in frozen_fields)
                self.assertTrue("R1_BLOCK" in frozen_fields)
                self.assertTrue("num_warps" in frozen_fields)
            else:
                self.assertTrue(len(frozen_fields) == 0)

            self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()
