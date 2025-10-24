# Owner(s): ["module: inductor"]

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import same
from torch._inductor import metrics, utils
from torch._inductor.test_case import run_tests, TestCase
from torch.testing import FileCheck
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

    @inductor_config.patch(split_reductions=False)
    def test_skip_due_to_non_persistent_reduction(self):
        """
        We only generate mix order reduction if one of the reduction is
        persistent reduction.
        """

        def f(x):
            return x.sum(dim=1), x.sum(dim=0)

        x = torch.randn(32768, 2048, device=GPU_TYPE)
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
    @parametrize("shape", ((32768, 768), (32769, 768)))
    @inductor_config.patch(split_reductions=False)
    def test_mix_order_reduction(self, name, swap, shape):
        def f(x):
            if swap:
                return reduction_fn(x, dim=0), reduction_fn(x, dim=1)
            else:
                return reduction_fn(x, dim=1), reduction_fn(x, dim=0)

        reduction_fn = getattr(torch, name)
        M, N = shape
        dtype = torch.float
        x = torch.randn(M, N, dtype=dtype, device=GPU_TYPE)

        opt_f = torch.compile(f)

        ref = f(x)
        act = opt_f(x)

        self.assertTrue(same(ref, act, tol=1e-3), f"ref:\n{ref}\nact:\n{act}")

        expected_num_kernel = 1 + (not inductor_config.triton.mix_order_reduction)
        if name == "mean" and inductor_config.triton.mix_order_reduction:
            # for mean we generate one more kernel to do the division
            # this kernel should be very cheap since tensor size is small
            expected_num_kernel = 2
        self.assertEqual(
            expected_num_kernel,
            metrics.generated_kernel_count,
        )

    @inductor_config.patch(split_reductions=False)
    def test_multi_workspace_allocation(self):
        def f(x, y):
            return x.sum(dim=0), x.sum(dim=1), y.sum(dim=0), y.sum(dim=1)

        x = torch.randn(128 * 15, 128, device=GPU_TYPE)
        y = torch.randn(256 * 15, 256, device=GPU_TYPE)

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
    @parametrize("shape", ((32768, 768), (32769, 768)))
    @inductor_config.patch(split_reductions=False)
    def test_rms_norm_bwd(self, wdtype, shape):
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

        opt_f = torch.compile(f)

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        expected_num_kernel = 1 + (not inductor_config.triton.mix_order_reduction)
        if wdtype == torch.bfloat16 and inductor_config.triton.mix_order_reduction:
            # one extra kernel for downcasting
            expected_num_kernel = 2
        FileCheck().check_count(
            "@triton.jit",
            expected_num_kernel,
            exactly=True,
        ).run(bwd_wrapper)

    @parametrize(
        "wbdtype",
        [
            torch.bfloat16,  # extra down cast for dw/db is needed
            torch.float,
        ],
    )
    @parametrize("shape", ((32768, 768), (32769, 768)))
    @inductor_config.patch(split_reductions=False)
    def test_layer_norm_bwd_with_bias(self, wbdtype, shape):
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

        opt_f = torch.compile(f)

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        expected_num_kernel = 1 + (not inductor_config.triton.mix_order_reduction)
        if wbdtype == torch.bfloat16 and inductor_config.triton.mix_order_reduction:
            # one extra kernel for downcasting
            expected_num_kernel = 2
        FileCheck().check_count(
            "@triton.jit",
            expected_num_kernel,
            exactly=True,
        ).run(bwd_wrapper)

    @parametrize("shape", ((32768, 768), (32769, 768)))
    @inductor_config.patch(split_reductions=False)
    def test_layer_norm_bwd_no_bias(self, shape):
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

        opt_f = torch.compile(f)

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        FileCheck().check_count(
            "@triton.jit",
            1 + (not inductor_config.triton.mix_order_reduction),
            exactly=True,
        ).run(bwd_wrapper)


@inductor_config.patch(
    "triton.mix_order_reduction", not inductor_config.triton.mix_order_reduction
)
class NoMixOrderReductionTest(MixOrderReductionTest):
    pass


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
