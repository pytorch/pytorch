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

    def test_use_deterministic_algorithsm(self):
        old_val = torch.are_deterministic_algorithms_enabled()
        try:
            for new_val in [True, False, True]:
                torch.use_deterministic_algorithms(new_val, warn_only=True)
                self.assertEqual(inductor_config.deterministic, new_val)
        finally:
            torch.use_deterministic_algorithms(old_val, warn_only=True)

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

            out = foo(inp)
            self.assertEqual(out, inp.sum(dim=-1))

            if deterministic:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] == 0)
            else:
                self.assertTrue(counters["inductor"]["coordesc_tuning_bench"] > 0)


@instantiate_parametrized_tests
class EagerConsistencyTest(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._exit_stack = contextlib.ExitStack()
        self._exit_stack.enter_context(fresh_cache())

    def tearDown(self) -> None:
        self._exit_stack.close()
        super().tearDown()

    @staticmethod
    def compute_fn_and_grads(fn, args, grad):
        args = [a.detach().requires_grad_(True) for a in args]
        out = fn(*args)
        out.backward(grad.detach())
        return out.detach(), [a.grad.detach() for a in args]

    def test_rms_norm_eager_can_find_orders(self):
        B = 128
        H = 256

        from torch._inductor._numeric_utils import find_order, fma, ordered_fma_sum

        def model_fwd(x, w, order):
            sum = ordered_fma_sum(x, x, dim=-1, keepdim=True, order=order)
            rsqrt = torch.rsqrt(sum / x.shape[-1] + 1e-5)
            return w * (x * rsqrt)

        fail_at_nnz_fwd, fwd_order = find_order(
            lambda x, w: torch.nn.functional.rms_norm(x, [x.shape[-1]], w, 1e-5),
            model_fwd,
            torch.float32,
            [(B, H), (H,)],
            [-1, -1],
        )

        self.assertTrue(fail_at_nnz_fwd is None)

        def model_bwd_input(x, w, g, order):
            sum = ordered_fma_sum(x, x, dim=-1, keepdim=True, order=fwd_order)
            rsqrt = torch.rsqrt(sum / x.shape[-1] + 1e-5)
            N = x.shape[-1]
            sum_val = ordered_fma_sum(
                (w * g) * x, rsqrt, dim=-1, keepdim=True, order=order
            )
            return ((1 / N) * rsqrt) * fma(-rsqrt * x * sum_val, N * w, g)

        def ref_bwd_input(x, w, g):
            x = x.detach().requires_grad_(True)
            o = torch.nn.functional.rms_norm(x, [x.shape[-1]], w, 1e-5)
            o.backward(g)
            return x.grad.detach()

        fail_at_nnz_bwd_input, _ = find_order(
            ref_bwd_input,
            model_bwd_input,
            torch.float32,
            [(B, H), (H,), (B, H)],
            [-1, -1, -1],
        )

        self.assertTrue(fail_at_nnz_bwd_input is None)

        def model_bwd_weight(x, w, g, order):
            sum = ordered_fma_sum(x, x, dim=-1, keepdim=True, order=fwd_order)
            rsqrt = torch.rsqrt(sum / x.shape[-1] + 1e-5)
            return ordered_fma_sum(x * g, rsqrt, dim=0, keepdim=False, order=order)

        def ref_bwd_weight(x, w, g):
            w = w.detach().requires_grad_(True)
            o = torch.nn.functional.rms_norm(x, [x.shape[-1]], w, 1e-5)
            o.backward(g)
            return w.grad.detach()

        fail_at_nnz_bwd_weight, _ = find_order(
            ref_bwd_weight,
            model_bwd_weight,
            torch.float32,
            [(B, H), (H,), (B, H)],
            [0, None, 0],
        )

        self.assertTrue(fail_at_nnz_bwd_weight is None)

    @inductor_config.patch({"triton.match_eager_rms_norm": True})
    def test_rms_norm_eager_equals_compile_decomp(self):
        B = 128
        H = 256
        dtype = torch.float32
        fn = lambda x, w: torch.nn.functional.rms_norm(x, [H], w)
        x = torch.randn(B, H, dtype=dtype, device=GPU_TYPE)
        w = torch.randn(H, dtype=dtype, device=GPU_TYPE)
        do = torch.randn(B, H, dtype=dtype, device=GPU_TYPE)
        args = (x, w)
        eager = self.compute_fn_and_grads(fn, args, do)
        compile = self.compute_fn_and_grads(torch.compile(fn), args, do)
        self.assertIdentical(eager, compile)

    def test_rms_norm_eager_equals_compile_raw(self):
        B = 128
        H = 256
        dtype = torch.float32
        eps = 1e-5
        fn = lambda x, w: torch.nn.functional.rms_norm(x, [H], w, eps)
        x = torch.randn(B, H, dtype=dtype, device=GPU_TYPE)
        w = torch.randn(H, dtype=dtype, device=GPU_TYPE)
        do = torch.randn(B, H, dtype=dtype, device=GPU_TYPE)
        args = (x, w)
        eager = self.compute_fn_and_grads(fn, args, do)
        out, r = torch._inductor.fx_passes.post_grad._fused_rms_norm_eager_decomp(
            x, [H], w, eps
        )
        dx, dw = (
            torch._inductor.fx_passes.post_grad._fused_rms_norm_backward_eager_decomp(
                do, x, [H], r, w, [True, True]
            )
        )
        self.assertIdentical(eager[0], out)
        self.assertIdentical(eager[1][0], dx)
        self.assertIdentical(eager[1][1], dw)


if __name__ == "__main__":
    if HAS_CUDA_AND_TRITON:
        run_tests()
