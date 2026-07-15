# Owner(s): ["module: higher order operators"]

import torch
import torch._dynamo
import torch._inductor
import torch._inductor.metrics as metrics
from torch._higher_order_ops import fuse_or_err
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestFuseOrErr(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        metrics.reset()

    def tearDown(self):
        super().tearDown()
        metrics.reset()

    @requires_gpu
    def test_single_kernel_pointwise(self):
        @fuse_or_err
        def region(x):
            return x.sin().cos() + 1

        def fn(x):
            return region(x)

        x = torch.randn(1024, device=GPU_TYPE)
        ref = x.sin().cos() + 1
        fn_c = torch.compile(fn, backend="inductor", fullgraph=True)
        res, _ = run_and_get_code(fn_c, x)
        self.assertEqual(ref, res)
        # The whole region collapsed into a single Triton kernel.
        self.assertEqual(metrics.generated_kernel_count, 1)

    @requires_gpu
    def test_captures_free_variables(self):
        # A closure that captures a free tensor variable still fuses to one
        # kernel (the free var is lifted to an operand by Dynamo).
        bias = torch.randn(1024, device=GPU_TYPE)

        def fn(x):
            return fuse_or_err(lambda a: a.sin() + bias)(x)

        x = torch.randn(1024, device=GPU_TYPE)
        ref = x.sin() + bias
        fn_c = torch.compile(fn, backend="inductor", fullgraph=True)
        res, _ = run_and_get_code(fn_c, x)
        self.assertEqual(ref, res)
        self.assertEqual(metrics.generated_kernel_count, 1)

    def _run_extern_region(self, device):
        w = torch.randn(64, 64, device=device)

        def fn(x):
            # torch.mm lowers to an extern kernel that can never fuse with the
            # surrounding pointwise sin, so the region cannot be one kernel.
            return fuse_or_err(lambda a: (a @ w).sin())(x)

        x = torch.randn(64, 64, device=device)
        torch.compile(fn, backend="inductor", fullgraph=True)(x)

    def test_errors_when_region_does_not_fuse(self):
        device = GPU_TYPE if torch.cuda.is_available() else "cpu"
        with self.assertRaisesRegex(RuntimeError, r"did not fuse into a single kernel"):
            self._run_extern_region(device)

    def test_error_message_surfaces_reason(self):
        device = GPU_TYPE if torch.cuda.is_available() else "cpu"
        try:
            self._run_extern_region(device)
            self.fail("expected RuntimeError")
        except RuntimeError as e:
            msg = str(e)
        self.assertIn("separate kernels", msg)
        # The offending extern op is named and flagged.
        self.assertIn("mm", msg)
        self.assertIn("extern", msg)

    @requires_gpu
    def test_backward_default_not_enforced(self):
        # Default fuse_backward=False: forward is checked, backward is not.
        def fn(x):
            return fuse_or_err(lambda a: a.sin().cos())(x).sum()

        x = torch.randn(1024, device=GPU_TYPE, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        torch.compile(fn, backend="inductor", fullgraph=True)(x).backward()
        (xr.sin().cos()).sum().backward()
        self.assertEqual(x.grad, xr.grad)

    @requires_gpu
    def test_backward_fuse_enforced(self):
        # fuse_backward=True: the backward region is also required to fuse; a
        # pointwise region fuses in both directions, so this succeeds.
        def fn(x):
            return fuse_or_err(lambda a: a.sin().cos(), fuse_backward=True)(x).sum()

        x = torch.randn(1024, device=GPU_TYPE, requires_grad=True)
        xr = x.detach().clone().requires_grad_(True)
        torch.compile(fn, backend="inductor", fullgraph=True)(x).backward()
        (xr.sin().cos()).sum().backward()
        self.assertEqual(x.grad, xr.grad)

    def _enforce_flags(self, fuse_backward):
        # Compiles a differentiable region and returns the _enforce_fusion flag
        # carried by the fuse_or_err node in the (forward, backward) graphs.
        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        fw_graphs = []
        bw_graphs = []

        def fw_compiler(gm, _):
            fw_graphs.append(gm)
            return make_boxed_func(gm.forward)

        def bw_compiler(gm, _):
            bw_graphs.append(gm)
            return make_boxed_func(gm.forward)

        def fn(x):
            return fuse_or_err(lambda a: a.sin().cos(), fuse_backward=fuse_backward)(
                x
            ).sum()

        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        x = torch.randn(64, requires_grad=True)
        torch._dynamo.reset()
        torch.compile(fn, backend=backend, fullgraph=True)(x).backward()

        def enforce_flag(gms):
            for gm in gms:
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target is torch.ops.higher_order.fuse_or_err
                    ):
                        return node.kwargs["_enforce_fusion"]
            return None

        return enforce_flag(fw_graphs), enforce_flag(bw_graphs)

    def test_forward_always_enforced_backward_gated(self):
        # Forward is always enforced; backward is enforced iff fuse_backward.
        fwd, bwd = self._enforce_flags(fuse_backward=False)
        self.assertTrue(fwd)
        self.assertFalse(bwd)

        fwd, bwd = self._enforce_flags(fuse_backward=True)
        self.assertTrue(fwd)
        self.assertTrue(bwd)

    def test_eager_runs_closure(self):
        # Outside torch.compile the closure just runs, with no fusion check.
        called = {}

        def region(x):
            called["yes"] = True
            return x.sin() + 1

        x = torch.randn(8)
        out = fuse_or_err(region)(x)
        self.assertEqual(out, x.sin() + 1)
        self.assertTrue(called["yes"])

    def test_eager_multi_input_autograd(self):
        # Eager execution calls the closure directly, so multi-input autograd
        # works (it does not route backward through the HOP).
        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        fuse_or_err(lambda a, b: a + b)(x, y).sum().backward()
        self.assertEqual(x.grad, torch.ones_like(x))
        self.assertEqual(y.grad, torch.ones_like(y))

    def test_eager_accepts_kwargs(self):
        # In eager, keyword arguments to the wrapped region are passed through.
        x = torch.randn(4)
        out = fuse_or_err(lambda a, b=1: a + b)(x, b=2)
        self.assertEqual(out, x + 2)

    def test_kwargs_under_compile_raise(self):
        def fn(x):
            return fuse_or_err(lambda a, b: a + b)(x, b=x)

        with self.assertRaisesRegex(RuntimeError, r"keyword arguments"):
            torch.compile(fn, backend="inductor", fullgraph=True)(torch.randn(8))

    @requires_gpu
    def test_upstream_lazy_operand_not_miscounted(self):
        # A region whose op realizes an upstream lazy operand as a side effect
        # must not count that operand as a region op (regression: false positive).
        def fn(x):
            y = x.sin()  # upstream lazy pointwise, becomes an operand
            return fuse_or_err(lambda a: a @ a)(y)  # region is a single extern mm

        x = torch.randn(64, 64, device=GPU_TYPE)
        ref = (x.sin()) @ (x.sin())
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x)
        self.assertEqual(ref, res)

    @requires_gpu
    def test_combo_kernel_counts_as_one(self):
        # With combo kernels, independent pointwise ops merge into one launched
        # kernel; the check runs after combo formation so this must not error.
        import torch._inductor.config as inductor_config

        def fn(x, y):
            return fuse_or_err(lambda a, b: (a.sin(), b.cos()))(x, y)

        x = torch.randn(128, device=GPU_TYPE)
        y = torch.randn(128, device=GPU_TYPE)
        ref = (x.sin(), y.cos())
        with inductor_config.patch(combo_kernels=True, benchmark_combo_kernel=False):
            res = torch.compile(fn, backend="inductor", fullgraph=True)(x, y)
        self.assertEqual(ref, res)

    @requires_gpu
    def test_zero_kernel_region_passes(self):
        # "at most one kernel": a region that materializes no kernel trivially
        # passes.
        def fn(x):
            return fuse_or_err(lambda a: a.size(0))(x)

        res = torch.compile(fn, backend="inductor", fullgraph=True)(
            torch.randn(8, device=GPU_TYPE)
        )
        self.assertEqual(res, 8)


if __name__ == "__main__":
    run_tests()
