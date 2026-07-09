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


if __name__ == "__main__":
    run_tests()
