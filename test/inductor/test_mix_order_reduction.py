# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import same
from torch._inductor import metrics, utils
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestBase(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()

    def check_numeric(self, f, args, tol=1e-3):
        ref = f(*args)
        act = torch.compile(f)(*args)
        self.assertTrue(same(ref, act, tol=tol))


class SkipPatternTest(TestBase):
    """
    Illustate the cases that we skip mix-order reduction. We skip in cases
    like when the outer reduction is followed by a pointwise that load
    the un-reduced tensor.
    """

    @inductor_config.patch(split_reductions=False)
    def test_dimension_too_close(self):
        """
        Skip if the two reduction size are too close.
        We require one reduction dimension to be much larger so we can split
        that dimension and make it efficient.
        """

        def f(x):
            out1 = x.sum(dim=1)
            out2 = x.sum(dim=0)
            return out1, out2

        x = torch.randn(768, 768, device=GPU_TYPE)
        torch.compile(f)(x)
        self.assertEqual(2, metrics.generated_kernel_count)

    @inductor_config.patch(split_reductions=False)
    def test_skip_if_outer_reduction_followed_by_full_pointwise(self):
        """
        Skip for now if the outer reduction is followed by a pointwise node
        accessing the original tensor. Accessing the reduced tensor is fine
        (e.g. to support torch.mean).
        """

        def f(x):
            out1 = x.sum(dim=1)
            out2 = x.sum(dim=0, keepdim=True) + x
            return out1, out2

        x = torch.randn(32768, 768, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.assertEqual(0, metrics.codegen_mix_order_reduction)


@instantiate_parametrized_tests
class MixOrderReductionTest(TestBase):
    @parametrize(
        "name",
        [
            "sum",
            "prod",
            "mean",
        ],
    )
    @parametrize("swap", (False, True))
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768), (32, 1024, 768)))
    def test_mix_order_reduction(self, name, swap, split_reductions, shape):
        # torch.prod does not accept tuple for dim argument
        if name == "prod" and len(shape) == 3:
            self.skipTest("Invalid combination")

        def f(x):
            def outer_red():
                if len(shape) == 3:
                    return reduction_fn(x, dim=(0, 1))
                else:
                    assert len(shape) == 2
                    return reduction_fn(x, dim=0)

            if swap:
                return outer_red(), reduction_fn(x, dim=-1)
            else:
                return reduction_fn(x, dim=-1), outer_red()

        reduction_fn = getattr(torch, name)
        dtype = torch.float
        x = torch.randn(shape, dtype=dtype, device=GPU_TYPE)

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = f(x)
        act = opt_f(x)

        self.assertTrue(same(ref, act, tol=1e-3), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @inductor_config.patch(coordinate_descent_tuning=True)
    def test_XBLOCK_coordest_tuning(self):
        """
        We should skip XBLOCK coordinate descent tuning for
        mix order reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=-1), x.sum(dim=0)

        x = torch.randn(32768, 256, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.assertEqual(metrics.codegen_mix_order_reduction, 1)

    @inductor_config.patch(unroll_reductions_threshold=1)
    def test_3layer_split_reduction(self):
        """
        Use a larger M and smaller N to trigger a 3 layer split reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=-1), x.sum(dim=0)

        x = torch.randn(32768 * 256, 2, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        # We don't do mix order reduction for split redutions
        # with more than 2 layers
        self.assertEqual(metrics.codegen_mix_order_reduction, 0)

    def test_independent_split_size(self):
        """
        Make sure mix order reduction can pick the split size it wants
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=-1), x.sum(dim=0)

        def check_one_split_size(split_size):
            torch._dynamo.reset()

            with inductor_config.patch(
                "triton.mix_order_reduction_split_size", split_size
            ):
                self.check_numeric(f, (x,))
                self.assertEqual(
                    inductor_config.triton.mix_order_reduction,
                    metrics.codegen_mix_order_reduction,
                )

                _, (code,) = utils.run_and_get_code(torch.compile(f), x)
                self.assertTrue(f"'RSPLIT_SIZE': {split_size}" in code)

        x = torch.randn(32768, 768, dtype=torch.float, device=GPU_TYPE)

        check_one_split_size(8)
        check_one_split_size(16)

    @inductor_config.patch(split_reductions=False)
    def test_non_contiguous_input(self):
        def f(x):
            return x.sum(dim=-1), x.sum(dim=[0, 1])

        x = torch.randn(1024, 32, 768, dtype=torch.float, device=GPU_TYPE).permute(
            1, 0, 2
        )
        self.check_numeric(f, (x,))
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @inductor_config.patch(split_reductions=False)
    def test_multi_workspace_allocation(self):
        def f(x, y):
            return x.sum(dim=0), x.sum(dim=1), y.sum(dim=0), y.sum(dim=1)

        x = torch.randn(4096, 32, device=GPU_TYPE)
        y = torch.randn(4098, 34, device=GPU_TYPE)

        self.check_numeric(f, (x, y))
        expected_mix_order_reduction = (
            0 if not inductor_config.triton.mix_order_reduction else 2
        )
        self.assertEqual(
            expected_mix_order_reduction, metrics.codegen_mix_order_reduction
        )

    @parametrize(
        "wdtype",
        [
            torch.bfloat16,  # extra down cast for dw is needed
            torch.float,
        ],
    )
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 2048), (32768, 768), (32769, 768)))
    def test_rms_norm_bwd(self, wdtype, split_reductions, shape):
        def f(x, w, eps):
            orig_dtype = x.dtype

            x = x.float()
            rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
            y = (x * rsqrt[:, None] * w).to(dtype=orig_dtype)
            return y

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        torch.manual_seed(1337)

        # M, N = 1152 * 500, 384
        M, N = shape
        x = torch.randn(M, N, dtype=torch.bfloat16, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize(
        "wbdtype",
        [
            torch.bfloat16,  # extra down cast for dw/db is needed
            torch.float,
        ],
    )
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768)))
    def test_layer_norm_bwd_with_bias(self, wbdtype, split_reductions, shape):
        def f(x, w, b, eps):
            return F.layer_norm(x, x.shape[-1:], w.float(), b.float(), eps)

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            b.grad = None
            out = f(x, w, b, eps)
            out.backward(dy)
            return x.grad, w.grad, b.grad

        # M, N = 1152 * 500, 384
        M, N = shape
        xdtype = torch.float
        x = torch.randn(M, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        b = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768)))
    def test_layer_norm_bwd_no_bias(self, split_reductions, shape):
        def f(x, w, eps):
            return F.layer_norm(x, x.shape[-1:], w, bias=None, eps=eps)

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        # M, N = 1152 * 500, 384
        M, N = shape
        xdtype = torch.float
        wbdtype = torch.float
        x = torch.randn(M, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )


@inductor_config.patch(
    "triton.mix_order_reduction", not inductor_config.triton.mix_order_reduction
)
class NoMixOrderReductionTest(MixOrderReductionTest):
    pass


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
