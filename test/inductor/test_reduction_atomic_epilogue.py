# Owner(s): ["module: inductor"]
import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


def _pointwise_then_amax(x, y):
    s = x * y + 1.0
    return s.abs().amax(), s * 2.0


@instantiate_parametrized_tests
class ReductionAtomicEpilogueTest(TestCase):
    def _compile(self, fn, *args):
        torch._dynamo.reset()
        return run_and_get_code(torch.compile(fn), *args)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    @parametrize("op", ["amax", "amin"])
    def test_fused_into_producer(self, op):
        def fn(x, y):
            s = x * y + 1.0
            return getattr(s, op)(), s * 2.0

        x = torch.randn(1 << 20, device=GPU_TYPE)
        y = torch.randn(1 << 20, device=GPU_TYPE)
        (red, pw), (code,) = self._compile(fn, x, y)
        ref_red, ref_pw = fn(x, y)
        self.assertEqual(red, ref_red, atol=0, rtol=0)
        self.assertEqual(pw, ref_pw)
        atomic = "tl.atomic_max" if op == "amax" else "tl.atomic_min"
        FileCheck().check(atomic).run(code)
        self.assertNotIn("triton_red_", code)
        self.assertNotIn("triton_per_", code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    def test_nan_and_inf(self):
        x = torch.randn(4097, device=GPU_TYPE)
        y = torch.randn(4097, device=GPU_TYPE)
        x[123] = float("inf")
        red, _ = self._compile(_pointwise_then_amax, x, y)[0]
        self.assertEqual(red, _pointwise_then_amax(x, y)[0])
        x[3000] = float("nan")
        red, _ = self._compile(_pointwise_then_amax, x, y)[0]
        self.assertTrue(torch.isnan(red))

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    @parametrize("op", ["amax", "amin"])
    def test_negative_nan(self, op):
        def fn(x):
            s = -x
            return getattr(s, op)(), s

        x = torch.randn(5000, device=GPU_TYPE)
        x[17] = float("nan")
        (red, _), (code,) = self._compile(fn, x)
        self.assertTrue(torch.isnan(red))
        FileCheck().check("tl.atomic_m").run(code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    def test_dynamic_shapes(self):
        def fn(x):
            s = x * 2.0
            return s.amax(), s

        compiled = torch.compile(fn, dynamic=True)
        for n in (1000, 4097, 70001):
            x = torch.randn(n, device=GPU_TYPE)
            red, s = compiled(x)
            ref_red, ref_s = fn(x)
            self.assertEqual(red, ref_red, atol=0, rtol=0)
            self.assertEqual(s, ref_s)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    @parametrize("n", [1, 7, 4096])
    def test_small_and_negative_zero(self, n):
        def fn(x):
            s = x * 2.0
            return s.amax(), s

        x = torch.full((n,), -0.0, device=GPU_TYPE)
        (out, _), (code,) = self._compile(fn, x)
        self.assertEqual(out, fn(x)[0], atol=0, rtol=0)
        self.assertTrue(torch.signbit(out))
        FileCheck().check("tl.atomic_max").run(code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    @parametrize("hidden", [15, 5000])
    def test_fused_into_reduction_kernel(self, hidden):
        # The producer reads a row reduction, so the atomic lands in a reduction
        # kernel and must respect the reduction mask.
        def fn(x):
            y = torch.nn.functional.layer_norm(x, [hidden], eps=1e-5)
            return y.abs().amax(), y

        x = torch.rand(3, hidden, device=GPU_TYPE)
        (out, y), (code,) = self._compile(fn, x)
        ref_out, ref_y = fn(x)
        self.assertEqual(out, ref_out, atol=1e-5, rtol=1e-5)
        self.assertEqual(y, ref_y, atol=1e-5, rtol=1e-5)
        FileCheck().check("tl.atomic_max").run(code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    def test_non_fp32_keeps_reduction(self):
        x = torch.randn(1 << 16, device=GPU_TYPE, dtype=torch.bfloat16)
        y = torch.randn(1 << 16, device=GPU_TYPE, dtype=torch.bfloat16)
        (red, _), (code,) = self._compile(_pointwise_then_amax, x, y)
        self.assertEqual(red, _pointwise_then_amax(x, y)[0])
        self.assertNotIn("tl.atomic_max", code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    def test_keepdim_and_dims(self):
        def fn(x):
            s = x * 3.0
            return s.amax(dim=(0, -1), keepdim=True), s

        x = torch.randn(33, 517, device=GPU_TYPE)
        (out, _), (code,) = self._compile(fn, x)
        self.assertEqual(out, fn(x)[0], atol=0, rtol=0)
        self.assertEqual(out.shape, (1, 1))
        FileCheck().check("tl.atomic_max").run(code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    def test_deterministic(self):
        x = torch.randn(3 << 20, device=GPU_TYPE)
        y = torch.randn(3 << 20, device=GPU_TYPE)
        compiled = torch.compile(_pointwise_then_amax)
        first = compiled(x, y)[0]
        for _ in range(3):
            self.assertEqual(compiled(x, y)[0], first, atol=0, rtol=0)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    def test_sole_consumer_keeps_reduction(self):
        # Nothing else reads the producer, so the plain reduction already fuses
        # it and the atomics would only add kernels.
        def fn(x):
            return (x * 2.0).abs().amax()

        x = torch.randn(1 << 16, device=GPU_TYPE)
        out, (code,) = self._compile(fn, x)
        self.assertEqual(out, fn(x), atol=0, rtol=0)
        self.assertNotIn("tl.atomic_max", code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True)
    def test_unfused_input_keeps_reduction(self):
        def fn(x):
            return x.amax()

        x = torch.randn(1 << 16, device=GPU_TYPE)
        out, (code,) = self._compile(fn, x)
        self.assertEqual(out, fn(x))
        self.assertNotIn("tl.atomic_max", code)

    @requires_gpu()
    def test_disabled_by_default(self):
        x = torch.randn(1 << 16, device=GPU_TYPE)
        y = torch.randn(1 << 16, device=GPU_TYPE)
        _, (code,) = self._compile(_pointwise_then_amax, x, y)
        self.assertNotIn("tl.atomic_max", code)

    @requires_gpu()
    @inductor_config.patch(reduction_atomic_epilogue=True, combo_kernels=True)
    def test_foreach_clip_pattern(self):
        # Optimizer-style: foreach update, per-tensor max-norm of the update, scaled
        # apply. The amax must land in the foreach kernel, with the larger block.
        def fn(ps, gs):
            upd = torch._foreach_mul(gs, 0.1)
            norms = [u.abs().amax() for u in upd]
            for p, u, n in zip(ps, upd, norms):
                p.sub_(u / torch.clamp(n, min=1.0))

        sizes = [4096 * 5 + 7, 1 << 16, 300, 1 << 20]
        ps = [torch.randn(s, device=GPU_TYPE) for s in sizes]
        gs = [torch.randn(s, device=GPU_TYPE) for s in sizes]
        ref = [p.clone() for p in ps]
        fn(ref, gs)
        _, (code,) = self._compile(fn, ps, gs)
        for p, r in zip(ps, ref):
            self.assertEqual(p, r)
        FileCheck().check("tl.atomic_max").run(code)
        self.assertIn(
            f"XBLOCK: tl.constexpr = {inductor_config.reduction_atomic_epilogue_min_block}",
            code,
        )


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
