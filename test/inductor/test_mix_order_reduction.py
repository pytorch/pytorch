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


@instantiate_parametrized_tests
class MixOrderReductionTest(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()

    @parametrize(
        "name",
        [
            "sum",
            "prod",
            "mean",
            # "max",
            # "min",
            # "amax",
            # "amin",
        ],
    )
    @parametrize("swap", (False, True))
    @inductor_config.patch(split_reductions=False)
    def test_mix_order_reduction(self, name, swap):
        def f(x):
            if swap:
                return reduction_fn(x, dim=0), reduction_fn(x, dim=1)
            else:
                return reduction_fn(x, dim=1), reduction_fn(x, dim=0)

        reduction_fn = getattr(torch, name)
        M, N = 32768, 768
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

    @parametrize(
        "wdtype",
        [
            torch.bfloat16,  # extra down cast for dw is needed
            torch.float,
        ],
    )
    @inductor_config.patch(split_reductions=False)
    def test_rms_norm_bwd(self, wdtype):
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
        M, N = 32768, 768
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
    @inductor_config.patch(split_reductions=False)
    def test_layer_norm_bwd_with_bias(self, wbdtype):
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
        M, N = 32768, 768
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

    @inductor_config.patch(split_reductions=False)
    def test_layer_norm_bwd_no_bias(self):
        def f(x, w, eps):
            return F.layer_norm(x, x.shape[-1:], w, bias=None, eps=eps)

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        # M, N = 1152 * 500, 384
        M, N = 32768, 768
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
