# Owner(s): ["module: inductor"]

from unittest import mock
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._dynamo.utils import same
from torch._inductor import metrics, utils
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.test_case import run_tests, TestCase
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    isRocmArchAnyOf,
    MI200_ARCH,
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
    @parametrize("dtype", (torch.bfloat16, torch.float))
    @parametrize("swap", (False, True))
    @parametrize("split_reductions", (False, True))
    @parametrize("shape", ((32768, 768), (32769, 768), (32, 1024, 768)))
    def test_mix_order_reduction(self, name, dtype, swap, split_reductions, shape):
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
        x = torch.randn(shape, dtype=dtype, device=GPU_TYPE)

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )

        ref = f(x)
        act = opt_f(x)
        tol = 1e-3 if dtype == torch.float else 1e-2
        self.assertTrue(same(ref, act, tol=tol), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    def test_xmask(self):
        """
        Make sure xmask is setup properly
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=0), x.sum(dim=1)

        M, N = 32768 + 1023, 768
        EXTRA_ROW = 1
        buf = torch.randn(M + EXTRA_ROW, N, device=GPU_TYPE)
        x = buf[:M, :]
        # make sure wrong xmask error loud if read excess elements
        buf[M:, :] = 1000000

        opt_f = torch.compile(
            f,
            options={
                "triton.mix_order_reduction_initial_xblock": 2,
            },
        )

        ref = f(x)
        act = opt_f(x)

        self.assertTrue(same(ref, act, tol=1e-3), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    def test_avoid_non_coalesced_access(self):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, y):
            return (x + y).sum(dim=-1), x.sum(dim=(0, 1))

        x = torch.randn(128, 256, 768, device=GPU_TYPE)
        y = torch.randn(128, 768, 256, device=GPU_TYPE).transpose(1, 2)
        self.check_numeric(f, (x, y))

        # we skip mix order reduction for such kernel since
        # we force XBLOCK to be 1, the access to tensor y would be
        # very inefficient.
        # TODO: support XBLOCK larger than 1. But in that case, we
        # would have bigger restriction on rnumel to avoid exploding
        # shared memory.
        self.assertEqual(metrics.codegen_mix_order_reduction, 0)

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

        x = torch.randn(32768 * 1024, 2, dtype=torch.float, device=GPU_TYPE)
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

        x = torch.randn(4096 * 64, 32, device=GPU_TYPE)
        y = torch.randn(4098 * 64, 34, device=GPU_TYPE)

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
    @parametrize(
        "shape", ((1000000, 256), (32768, 2048), (32768, 768), (32768 + 1023, 768))
    )
    @parametrize("max_autotune", (False, True))
    @parametrize("initial_xblock", (1, 2))
    @parametrize("add_1dim", (False, True))
    def test_rms_norm_bwd(
        self,
        wdtype,
        split_reductions,
        shape,
        max_autotune,
        initial_xblock,
        add_1dim,
    ):
        # max_autotune can be slow and cost resource, trim down the tests
        # for max autotune
        if max_autotune and not (
            wdtype == torch.bfloat16
            and not split_reductions
            and shape in ((32768, 768), (32769, 768))
            and initial_xblock == 1
            and inductor_config.triton.mix_order_reduction
        ):
            self.skipTest("Skip non-critical tests to save resources.")

        if shape != (1000000, 256) and add_1dim:
            self.skipTest("Skip non-critical tests to save resources.")

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
        if add_1dim:
            x = x[:, None, :]

        w = torch.randn(N, dtype=wdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
                "triton.mix_order_reduction_initial_xblock": initial_xblock,
                **(
                    {
                        "max_autotune": True,
                        "coordinate_descent_tuning": True,
                    }
                    if max_autotune
                    else {}
                ),
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

    @parametrize("dynamic_dims", ([0], [1], [0, 1]))
    def test_rms_norm_bwd_with_dynamic_shape(self, dynamic_dims):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, w, eps):
            return F.rms_norm(x, x.shape[-1:], weight=w, eps=eps)

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        M0, M1, N = 251, 223, 128
        wbdtype = torch.float
        xdtype = torch.float
        x = torch.randn(M0, M1, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        torch._dynamo.mark_dynamic(x, (0, 1))
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(
            f,
            options={
                "split_reductions": False,
            },
        )

        ref = fwd_bwd(f)
        act, (_, bwd_wrapper) = utils.run_and_get_code(fwd_bwd, opt_f)

        self.assertTrue(same(ref, act, tol=1e-2), f"ref:\n{ref}\nact:\n{act}")
        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    @parametrize("dynamic_dims", ([0], [1], [0, 1]))
    def test_layer_norm_bwd_with_dynamic_shape(self, dynamic_dims):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x, w, eps):
            return F.layer_norm(x, x.shape[-1:], weight=w, bias=None, eps=eps)

        def fwd_bwd(f):
            x.grad = None
            w.grad = None
            out = f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        M0, M1, N = 251, 223, 128
        wbdtype = torch.float
        xdtype = torch.float
        x = torch.randn(M0, M1, N, dtype=xdtype, device=GPU_TYPE, requires_grad=True)
        torch._dynamo.mark_dynamic(x, dynamic_dims)
        w = torch.randn(N, dtype=wbdtype, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(f)

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

    @parametrize("split_reductions", (False, True))
    @parametrize("dtype", [torch.bfloat16, torch.float])
    def test_rms_norm_sharing_weights(self, split_reductions, dtype):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        if dtype is torch.bfloat16 and isRocmArchAnyOf(MI200_ARCH):
            self.skipTest("Currently failing on rocm mi200")

        def f(xs, w, eps):
            ys = []
            for x in xs:
                ys.append(F.rms_norm(x, x.shape[-1:], w, eps=eps))
            return ys

        num_norm = 3
        M, N = 32768, 768
        xs = [
            torch.randn(M, N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
            for _ in range(num_norm)
        ]
        w = torch.randn(N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        dys = [torch.randn_like(xs[0]) for _ in range(num_norm)]
        eps = 1e-5

        ref = f(xs, w, eps)
        act = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )(xs, w, eps)
        ref_grads = torch.autograd.grad(ref, [*xs, w], dys)
        act_grads, (wrapper,) = utils.run_and_get_code(
            lambda: torch.autograd.grad(act, [*xs, w], dys)
        )
        # bfloat16 cause big numerical instability for grad_weight
        tol = 1e-3 if dtype == torch.float32 else 0.5
        self.assertTrue(same((ref, ref_grads), (act, act_grads), tol=tol))
        self.assertEqual(
            metrics.codegen_mix_order_reduction,
            num_norm,
        )

        # a single mix order reduction kernel get shared
        FileCheck().check_count("MixOrderReductionGrid", 1, exactly=True).run(wrapper)

    @parametrize("split_reductions", (False, True))
    @parametrize("dtype", [torch.bfloat16, torch.float])
    @parametrize("has_bias", [False, True])
    def test_layer_norm_sharing_weights(self, split_reductions, dtype, has_bias):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(xs, w, bias, eps):
            ys = []
            for x in xs:
                ys.append(F.layer_norm(x, x.shape[-1:], w, bias=bias, eps=eps))
            return ys

        num_norm = 3
        M, N = 32768, 768
        xs = [
            torch.randn(M, N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
            for _ in range(num_norm)
        ]
        w = torch.randn(N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
        b = (
            torch.randn(N, dtype=dtype, device=GPU_TYPE, requires_grad=True)
            if has_bias
            else None
        )
        dys = [torch.randn_like(xs[0]) for _ in range(num_norm)]
        eps = 1e-5

        ref = f(xs, w, b, eps)
        act = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )(xs, w, b, eps)

        inputs_for_grad = [*xs, w]
        if has_bias:
            inputs_for_grad.append(b)
        ref_grads = torch.autograd.grad(ref, inputs_for_grad, dys)
        act_grads, (wrapper,) = utils.run_and_get_code(
            lambda: torch.autograd.grad(act, inputs_for_grad, dys)
        )
        tol = 1e-3 if dtype == torch.float32 else 1e-2
        if GPU_TYPE == "xpu":
            tol = 1e-3 if dtype == torch.float32 else 2e-2
        self.assertTrue(same((ref, ref_grads[:-2]), (act, act_grads[:-2]), tol=tol))
        if dtype == torch.float32:
            # bfloat16 cause big numerical instability for grad_weight
            # and grad_bias
            torch.testing.assert_close(
                ref_grads[-2:], act_grads[-2:], atol=tol, rtol=tol
            )
        self.assertEqual(
            metrics.codegen_mix_order_reduction,
            num_norm,
        )

        # a single mix order reduction kernel get shared
        FileCheck().check_count("MixOrderReductionGrid", 1, exactly=True).run(wrapper)

    @inductor_config.patch(split_reductions=False)
    def test_dont_fuse_nodes_that_introduce_producer_consumer_rel(self):
        """
        The test constructs an inner reduction, an outer reduction and
        a pointwise kernel.

        The inner reduction and outer reduction will be fused first.
        We don't further fuse the pointwise kernel (with the inner reduction part)
        since that introduces producer/consumer relationship between
        the inner and outer reduction.
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            out1 = x.sum(dim=1)
            out2 = x.sum(dim=0, keepdim=True) + x
            return out1, out2

        x = torch.randn(32768, 768, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.assertEqual(1, metrics.codegen_mix_order_reduction)
        # two kernels
        # one is the mix-order reduction kernel
        # the other is the piontwise kernel
        self.assertTrue(2, metrics.generated_kernel_count)

    @patch("torch._inductor.scheduler.MixOrderReduction.get_numel_rnumel")
    @patch("torch._inductor.scheduler.MixOrderReduction.get_common_read")
    @patch("torch._inductor.scheduler.MixOrderReduction.has_mix_reduction_orders")
    def test_mix_order_reduction_non_strict_mode(
        self,
        mock_has_mix_reduction_orders: mock.Mock,
        mock_get_common_read: mock.Mock,
        mock_get_numel_rnumel: mock.Mock,
    ):
        """
        This tests whether we can skip some non-critical checks
        when non_strict mode is on
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        from torch._inductor.scheduler import BaseSchedulerNode

        mock_node_1 = mock.create_autospec(BaseSchedulerNode)
        mock_node_2 = mock.create_autospec(BaseSchedulerNode)

        mock_node_1.is_gpu.return_value = True
        mock_node_2.is_gpu.return_value = True

        mock_node_1.get_device.return_value.type = "cuda"
        mock_node_1.is_reduction.return_value = True
        mock_node_2.is_reduction.return_value = True

        from torch._inductor.utils import OrderedSet

        mock_node_1.ancestors = OrderedSet()
        mock_node_2.ancestors = OrderedSet()
        mock_node_1.get_operation_names.return_value = OrderedSet()
        mock_node_2.get_operation_names.return_value = OrderedSet()

        mock_has_mix_reduction_orders.return_value = True
        mock_get_common_read.return_value = "common_read"
        from sympy import Integer

        mock_get_numel_rnumel.return_value = (Integer(1), Integer(1))

        mock_node_1.read_writes = mock.Mock()
        mock_node_1.read_writes.reads = []

        # Create a dummy graph
        from torch._inductor.graph import GraphLowering
        from torch._inductor.virtualized import V
        from torch.fx.experimental.proxy_tensor import make_fx

        gm = make_fx(lambda: torch.zeros(2, 3))()
        graph = GraphLowering(gm)

        with (
            V.set_graph_handler(graph),
            inductor_config.patch(
                {"triton.mix_order_reduction_non_strict_mode": False}
            ),
        ):
            self.assertFalse(MixOrderReduction.can_fuse(mock_node_1, mock_node_2))
        with (
            V.set_graph_handler(graph),
            inductor_config.patch({"triton.mix_order_reduction_non_strict_mode": True}),
        ):
            self.assertTrue(MixOrderReduction.can_fuse(mock_node_1, mock_node_2))


@inductor_config.patch(
    "triton.mix_order_reduction", not inductor_config.triton.mix_order_reduction
)
class NoMixOrderReductionTest(MixOrderReductionTest):
    pass


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
