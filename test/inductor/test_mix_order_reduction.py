# Owner(s): ["module: inductor"]

from unittest import mock
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch import nn
from torch._dynamo.utils import same
from torch._inductor import metrics, utils
from torch._inductor.scheduler import MixOrderReduction
from torch._inductor.test_case import run_tests, TestCase
from torch.testing import FileCheck
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestBase(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()

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
                    assert len(shape) == 2  # noqa: S101
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

    @inductor_config.patch(split_reductions=False)
    def test_fuse_non_contiguous_pointwise(self):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        # Regression: mix-order reduction can appear valid pre-fusion, but a pointwise
        # fused into one side can change access patterns and break the contiguity
        # invariant. This test builds a reduction + pointwise path plus a second
        # reduction, matching the shape/ordering pattern seen in the E2E failure.

        def f(x):
            # First reduction (contiguous on its own).
            r1 = x.sum(dim=1)
            # Pointwise depends on both reduced and unreduced data, so fusing it
            # with the reduction can change access strides.
            y = r1 * x[:, 0]
            # Second reduction across a different dimension to trigger mix-order logic.
            r2 = x.sum(dim=0)
            return y, r2

        # Large, asymmetric shape encourages mix-order reduction heuristics.
        x = torch.randn(32768, 768, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))

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

        M = 32768 * 1024 if torch.version.hip is not None else 32768 * 256
        x = torch.randn(M, 2, dtype=torch.float, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        # We don't do mix order reduction for split redutions
        # with more than 2 layers
        self.assertEqual(
            metrics.codegen_mix_order_reduction,
            1
            if inductor_config.triton.cooperative_reductions
            or inductor_config.triton.force_cooperative_reductions
            else 0,
        )

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
    # The test OOM in CI sometimes. Ask for more memory to make it stable.
    @largeTensorTest("16GB", device=GPU_TYPE, inductor=True)
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

        # use float64 to compute ref_grads for precision
        # and cast back to original dtype
        xs_f64 = [x.to(torch.float64) for x in xs]
        w_f64 = w.to(torch.float64)
        dys_f64 = [dy.to(torch.float64) for dy in dys]
        ref_f64 = f(xs_f64, w_f64, eps)
        ref_grads_f64 = torch.autograd.grad(ref_f64, [*xs_f64, w_f64], dys_f64)
        ref_grads = [g.to(dtype) for g in ref_grads_f64]

        act = torch.compile(
            f,
            options={
                "split_reductions": split_reductions,
            },
        )(xs, w, eps)
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

    @patch("torch._inductor.scheduler.MixOrderReduction.is_split_reduction")
    @patch("torch._inductor.scheduler.MixOrderReduction.get_numel_rnumel")
    @patch("torch._inductor.scheduler.MixOrderReduction.get_common_read")
    @patch("torch._inductor.scheduler.MixOrderReduction.has_mix_reduction_orders")
    def test_mix_order_reduction_non_strict_mode(
        self,
        mock_has_mix_reduction_orders: mock.Mock,
        mock_get_common_read: mock.Mock,
        mock_get_numel_rnumel: mock.Mock,
        mock_is_split_reduction: mock.Mock,
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
        mock_is_split_reduction.return_value = False

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
            inductor_config.patch(
                {
                    "triton.mix_order_reduction_non_strict_mode": True,
                }
            ),
        ):
            self.assertTrue(MixOrderReduction.can_fuse(mock_node_1, mock_node_2))

    @inductor_config.patch({"triton.mix_order_reduction_non_strict_mode": True})
    def test_no_recompile(self):
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        def f(x):
            return x.sum(dim=1), x.sum(dim=0)

        x0 = torch.randn(2048, 1024, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x0, (0,))
        opt_f = torch.compile(f)

        ref = f(x0)
        act = opt_f(x0)

        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)
        self.assertEqual(metrics.codegen_mix_order_reduction, 1)

        opt_f(torch.randn(4096, 1024, device=GPU_TYPE))
        opt_f(torch.randn(512, 1024, device=GPU_TYPE))

        compile_metrics = torch._dynamo.utils._compilation_metrics
        self.assertEqual(len(compile_metrics), 1, "Don't recompile")

    @skipIfXpu(msg="https://github.com/intel/intel-xpu-backend-for-triton/issues/6398")
    def test_additive_rnumel(self):
        """
        Fix https://github.com/pytorch/pytorch/issues/176375
        """
        x = torch.randn(32768, 300, device=GPU_TYPE)
        y = torch.randn(32768, 200, device=GPU_TYPE)
        w = torch.randn(550, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 1)
        torch._dynamo.mark_dynamic(y, 1)

        def f(x, y, w):
            z = torch.cat((x, y), dim=1)

            # Slice w_pool to match dynamic dim. This avoids a guard that would
            # resolve s0+s1 to a concrete value (640), which is essential for
            # keeping rnumel symbolic in the generated code.
            w = w[: z.shape[-1]]  # [s0+s1], no concrete-equality guard
            z = z * w
            scale = z.sum()
            z = z + scale
            return z.sum(dim=0), z.sum(dim=1)

        ref = f(x, y, w)
        act = torch.compile(f)(x, y, w)

        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)

        self.assertEqual(
            inductor_config.triton.mix_order_reduction,
            metrics.codegen_mix_order_reduction,
        )

    def test_additive_num_splits(self):
        """
        When the `num_splits` is an additive expression, a pair of
        parenthesis is required.
        """
        torch.set_float32_matmul_precision("high")
        linear1 = nn.Linear(1000, 1000).to(GPU_TYPE)
        norm = nn.LayerNorm(1000).to(GPU_TYPE)

        def model(x):
            return norm(linear1(x[:, :-1].reshape(-1, 1000)))

        compiled_model = torch.compile(model)
        x = torch.randn(32, 200, 1000, device=GPU_TYPE)
        torch._dynamo.mark_dynamic(x, 1)
        compiled_model(x).sum().backward()

        act = linear1.weight.grad, linear1.bias.grad

        linear1.zero_grad()
        norm.zero_grad()
        model(x).sum().backward()
        ref = linear1.weight.grad, linear1.bias.grad

        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)

    @largeTensorTest("36GB", device=GPU_TYPE, inductor=True)
    def test_out_of_shared_memory(self):
        """
        Fix https://github.com/pytorch/pytorch/issues/175250
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        NUM_HEADS = 32
        NUM_KV_HEADS = 8
        HEAD_DIM = 128
        HIDDEN_SIZE = NUM_HEADS * HEAD_DIM * 2
        SEQ_LEN = 8192 * 2

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos, sin):
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
            return (q * cos) + (rotate_half(q) * sin), (k * cos) + (
                rotate_half(k) * sin
            )

        @torch.compile
        def forward(
            x,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            embed_norm,
            hidden_norm,
            cos,
            sin,
        ):
            batch, seq_len, _ = x.shape

            # Eagle3 first layer: split concatenated [embeds, hidden] input
            mid = x.shape[2] // 2
            embeds, hidden = x.split(mid, dim=-1)

            # Dual RMSNorm (pow, sum, div, mul in backward)
            embeds = embed_norm(embeds)
            hidden = hidden_norm(hidden)
            residual = hidden

            # Recombine for attention input (2 * HIDDEN_SIZE)
            x = torch.cat([embeds, hidden], dim=-1)

            # Adding a graph break here "fixes" the issue
            # by breaking up the fused op
            # torch._dynamo.graph_break()

            # Q/K/V projections from 2*hidden_size input
            q = q_proj(x).view(batch, seq_len, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = k_proj(x).view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
            v = v_proj(x).view(batch, seq_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)

            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            k = torch.repeat_interleave(k, NUM_HEADS // NUM_KV_HEADS, dim=1)
            v = torch.repeat_interleave(v, NUM_HEADS // NUM_KV_HEADS, dim=1)
            out = q.contiguous() @ k.contiguous().transpose(-2, -1) @ v.contiguous()

            out = out.transpose(1, 2).contiguous().reshape(batch, seq_len, -1)
            return o_proj(out) + residual

        # Layers
        embed_norm = nn.RMSNorm(HIDDEN_SIZE).to(GPU_TYPE)
        hidden_norm = nn.RMSNorm(HIDDEN_SIZE).to(GPU_TYPE)
        # Q/K/V project from 2*HIDDEN_SIZE (concatenated embeds + hidden)
        q_proj = nn.Linear(2 * HIDDEN_SIZE, NUM_HEADS * HEAD_DIM, bias=False).to(
            GPU_TYPE
        )
        k_proj = nn.Linear(2 * HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False).to(
            GPU_TYPE
        )
        v_proj = nn.Linear(2 * HIDDEN_SIZE, NUM_KV_HEADS * HEAD_DIM, bias=False).to(
            GPU_TYPE
        )
        o_proj = nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN_SIZE, bias=False).to(GPU_TYPE)

        # Block mask - simple causal only
        def causal_mask(_b, _h, q, kv):
            return q >= kv

        # Rotary embeddings (precomputed, no grad needed)
        inv_freq = 1.0 / (
            500000.0
            ** (
                torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=GPU_TYPE)
                / HEAD_DIM
            )
        )
        pos = torch.arange(1, SEQ_LEN + 1, dtype=torch.float32, device=GPU_TYPE)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        cos, sin = emb.cos(), emb.sin()

        # Input: 2*HIDDEN_SIZE to match split [embeds, hidden]
        x = torch.randn(
            1, SEQ_LEN, 2 * HIDDEN_SIZE, device=GPU_TYPE, requires_grad=True
        )

        out = forward(
            x,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            embed_norm,
            hidden_norm,
            cos,
            sin,
        )
        loss = out.sum()
        loss.backward()
        self.assertTrue(metrics.codegen_mix_order_reduction > 1)

    @inductor_config.patch("triton.mix_order_reduction", True)
    @inductor_config.patch("triton.mix_order_reduction_non_strict_mode", True)
    def test_dimension_refactoring_mismatch(self):
        """
        This reproduces an issue where `simplify_and_reorder()` produces a different
        dimension factorization than `_original_ranges` used during fusion decision.
        For example, fusion might see (13, 8472) but codegen sees (26, 4236) after
        the reduction split optimization adds a factor of 2 to the pointwise dimensions.

        We skip fusing split reductions for node1 in this case.
        """

        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        # Reproduce the RMSNorm backward pattern that triggered the bug.
        # The key is:
        # - Shape (M, N) = (13, 8472) where N=8472 is large enough to trigger split
        # - RMSNorm backward creates reductions along both dimensions
        # - The feature dimension reduction (8472) gets split with factor 2
        # - Mix order reduction tries to fuse these, but groups don't match after split
        def f(x, w, eps):
            orig_dtype = x.dtype
            x = x.float()
            # RMSNorm forward: y = x * rsqrt(mean(x^2) + eps) * w
            rsqrt = torch.rsqrt((x * x).sum(dim=-1) / x.shape[-1] + eps)
            y = (x * rsqrt[:, None] * w).to(dtype=orig_dtype)
            return y

        def fwd_bwd(compiled_f):
            x.grad = None
            w.grad = None
            out = compiled_f(x, w, eps)
            out.backward(dy)
            return x.grad, w.grad

        # Use the exact shape from the bug report: (13, 8472)
        # 8472 = 2 * 4236, so split with factor 2 gives sub-reductions of 4236
        M, N = 13, 8472
        x = torch.randn(M, N, dtype=torch.float32, device=GPU_TYPE, requires_grad=True)
        w = torch.randn(N, dtype=torch.float32, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)
        eps = 1e-5

        opt_f = torch.compile(f)

        ref = fwd_bwd(f)
        act = fwd_bwd(opt_f)
        torch.testing.assert_close(ref, act, atol=1e-3, rtol=1e-3)
        self.assertGreaterEqual(metrics.codegen_mix_order_reduction, 0)

    def test_keepdim_shape_mismatch(self):
        """
        Test that MixOrderReduction correctly handles keepdim=True reductions.

        This test reproduces a bug where the final reduction in MixOrderReduction
        generates `view(nsplit, rnumel).sum(dim=0)` which produces shape [rnumel],
        but the expected output should be [1, rnumel] when keepdim=True.

        The error manifests as:
        RuntimeError: Function CompiledFunctionBackward returned an invalid gradient
        at index N - got [2048] but expected shape compatible with [1, 2048]
        """
        if not inductor_config.triton.mix_order_reduction:
            self.skipTest("Mix order reduction not enabled")

        # Create a model that produces reductions with keepdim=True in the backward pass
        # This pattern is common in normalization layers like RMSNorm/LayerNorm
        class KeepDimReductionModel(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                # Using shape [1, hidden_size] to ensure keepdim=True in backward
                self.weight = nn.Parameter(torch.ones(1, hidden_size))
                self.bias = nn.Parameter(torch.zeros(1, hidden_size))

            def forward(self, x):
                # x: [batch, hidden_size]
                # Normalization-like operation that produces keepdim reductions in backward
                mean = x.mean(dim=-1, keepdim=True)
                var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
                x_norm = (x - mean) / (var + 1e-5).sqrt()
                return x_norm * self.weight + self.bias

        M, N = 32768, 2048  # Large batch to trigger mix order reduction
        model = KeepDimReductionModel(N).to(GPU_TYPE)

        x = torch.randn(M, N, dtype=torch.float32, device=GPU_TYPE, requires_grad=True)
        dy = torch.randn_like(x)

        def fwd_bwd(model, x, dy):
            x.grad = None
            model.zero_grad()
            out = model(x)
            out.backward(dy)
            return x.grad, model.weight.grad, model.bias.grad

        # Reference (eager)
        ref = fwd_bwd(model, x, dy)

        # Compiled with mix order reduction
        compiled_model = torch.compile(model)
        act = fwd_bwd(compiled_model, x, dy)

        # Verify numerical correctness
        self.assertTrue(same(ref, act, tol=1e-3), f"ref:\n{ref}\nact:\n{act}")

        # Verify mix order reduction was used
        self.assertGreater(
            metrics.codegen_mix_order_reduction,
            0,
            "Mix order reduction should be triggered",
        )


@inductor_config.patch(
    "triton.mix_order_reduction", not inductor_config.triton.mix_order_reduction
)
class NoMixOrderReductionTest(MixOrderReductionTest):
    pass


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
