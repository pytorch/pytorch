# Owner(s): ["module: inductor"]
from __future__ import annotations

import importlib.util
import re
import unittest
from unittest.mock import patch

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._inductor import config, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    HAS_HELION,
)


HAS_CUTE = importlib.util.find_spec("cutlass") is not None and (
    importlib.util.find_spec("cutlass.cute") is not None
)
SOURCE_RE = re.compile(r'helion_\w+_source = """\\\n(.*?)"""', re.DOTALL)
DOT_RE = re.compile(r"torch\.b?addb?mm\(")
BACKENDS = ["triton", "cute"]


def randn(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, device=GPU_TYPE, dtype=dtype)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
@instantiate_parametrized_tests
class TestHelionNativeMatmul(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        self.enterContext(
            config.patch(
                {
                    "helion.enabled": True,
                    "triton.native_matmul": True,
                    "helion.autotune_effort": "none",
                }
            )
        )

    def compile(
        self, fn, *args, expected=None, tol=None, backend="triton", helion=True
    ):
        """Run compiled `fn`, check it against FP32 eager, and return the Helion
        matmul kernel sources."""
        if backend == "cute" and not HAS_CUTE:
            self.skipTest("requires CuTeDSL")
        with (
            config.patch({"helion.backend": backend}),
            self.assertNoLogs("torch._inductor.scheduler", level="CRITICAL"),
        ):
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        if expected is None:
            low = (torch.float16, torch.bfloat16)
            expected = fn(*(a.float() if a.dtype in low else a for a in args))
            expected = pytree.tree_map(lambda e, r: e.to(r.dtype), expected, result)
        self.assertEqual(result, expected, atol=tol, rtol=tol)
        self.code = "\n".join(codes)
        sources = [s for s in SOURCE_RE.findall(self.code) if DOT_RE.search(s)]
        if helion:
            self.assertTrue(sources)
            self.assertNotRegex(self.code, r"extern_kernels\.\w*mm\w*\(")
            self.assertNotIn("tl.dot(", self.code)
        return sources

    def test_mm_source(self):
        (source,) = self.compile(
            lambda a, b, bias: torch.relu(a @ b + bias),
            randn(40, 72),
            randn(72, 136),
            randn(136),
        )
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([40, 136]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(72):
            tmp0 = in0[tile_y, tile_r]
            tmp1 = in1[tile_r, tile_x]
            acc = torch.addmm(acc, tmp0, tmp1)
        tmp2 = in2[tile_x].to(torch.float32)
        tmp3 = acc + tmp2
        tmp4 = torch.relu(tmp3)
        out0[tile_y, tile_x] = tmp4.to(torch.bfloat16)
""",
        )
        self.assertIn(
            "helion_fused_add_mm_relu_0 = PyCodeCache.load("
            "helion_fused_add_mm_relu_0_source).kernel",
            self.code,
        )
        self.assertIn(
            "helion_fused_add_mm_relu_0(arg0_1, arg1_1, arg2_1, buf1)", self.code
        )

    @parametrize("backend", BACKENDS)
    def test_bmm_epilogue(self, backend):
        (source,) = self.compile(
            lambda a, b, bias: torch.relu(torch.bmm(a, b)) + bias,
            randn(3, 40, 72),
            randn(3, 72, 136),
            randn(136),
            backend=backend,
        )
        self.assertIn("for tile_z, tile_y, tile_x in hl.tile([3, 40, 136]):", source)
        self.assertIn("tmp0 = in0[tile_z, tile_y, tile_r]", source)
        self.assertIn("tmp1 = in1[tile_z, tile_r, tile_x]", source)
        self.assertIn("acc = torch.baddbmm(acc, tmp0, tmp1)", source)
        self.assertIn("out0[tile_z, tile_y, tile_x] = ", source)

    def test_computed_operand(self):
        (source,) = self.compile(
            lambda a, b: (a + 1) @ (b - 2), randn(40, 72), randn(72, 136)
        )
        self.assertIn("in0[tile_y, tile_r]", source)
        self.assertIn("in1[tile_r, tile_x]", source)

    @parametrize("epilogue", [False, True])
    def test_horizontal(self, epilogue):
        def fn(a, b, c):
            first, second = a @ b, a @ c
            return (
                torch.relu(first - second + first * second)
                if epilogue
                else (first, second)
            )

        (source,) = self.compile(fn, randn(40, 72), randn(72, 136), randn(72, 136))
        self.assertIn("acc1 = hl.zeros(", source)
        self.assertIn("acc1 = torch.addmm(acc1, tmp0, ", source)
        self.assertEqual(source.count("in0[tile_y, tile_r]"), 1)
        self.assertIn("in2[tile_r, tile_x]", source)
        self.assertEqual("out1[" in source, not epilogue)

    def test_masked_epilogue_load(self):
        (source,) = self.compile(
            lambda a, b, first, second: a @ b + torch.cat((first, second)),
            randn(40, 72),
            randn(72, 136),
            randn(64),
            randn(72),
        )
        self.assertIn("hl.load(in2, [tile_x.index], extra_mask=", source)
        self.assertIn("hl.load(in3, [(-64) + tile_x.index], extra_mask=", source)

    @parametrize(
        "case",
        [
            "dynamic",
            "broadcast_rhs",
            "permuted_operand",
            "gathered_operand",
            "cpu_scalar",
        ],
    )
    def test_fallback(self, case):
        a, b = randn(32, 64), randn(64, 48)
        fn = lambda a, b: torch.relu(a @ b)  # noqa: E731
        args = [a, b]
        if case == "dynamic":
            torch._dynamo.mark_dynamic(a, 0)
        elif case == "broadcast_rhs":
            fn, args = lambda a, v: a @ v.expand(64, 48), [a, randn(64, 1)]
        elif case == "permuted_operand":
            fn = lambda a, b: a.permute(1, 0, 2).reshape(32, 64) @ b  # noqa: E731
            args = [randn(8, 32, 8), b]
        elif case == "gathered_operand":
            fn = lambda a, i, b: a[i] @ b  # noqa: E731
            args = [randn(80, 64), torch.randint(0, 80, (32,), device=GPU_TYPE), b]
        else:
            # The CPU scalar epilogue fuses into the matmul, so the kernel is Triton.
            fn = lambda a, b, c: a @ b + c  # noqa: E731
            args = [a, b, torch.tensor(2.0, device="cpu")]
        sources = self.compile(fn, *args, expected=fn(*args), tol=2e-2, helion=False)
        self.assertEqual(sources, [])

    def test_cudagraph_tree(self):
        def fn(a, b):
            return torch.relu((a + 1) @ b), a.sum(dim=-1)

        args = [randn(32, 64), randn(64, 48)]
        counters.clear()
        with config.patch({"triton.cudagraphs": True}):
            compiled = torch.compile(fn, fullgraph=True)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*args), fn(*args))
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
class TestHelionDefaults(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_default_config(self):
        import helion

        with (
            config.patch({"helion.enabled": True, "triton.native_matmul": True}),
            patch.object(helion.Kernel, "__call__", lambda self, *args: None),
        ):
            _, (code,) = run_and_get_code(
                torch.compile(lambda a, b: a @ b), randn(32, 64), randn(64, 48)
            )
        (source,) = [s for s in SOURCE_RE.findall(code) if DOT_RE.search(s)]
        self.assertIn('autotune_effort="quick"', source)
        self.assertNotRegex(code, r"extern_kernels\.\w*mm\w*\(")


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
@instantiate_parametrized_tests
class TestHelionReduction(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        self.enterContext(
            config.patch({"helion.enabled": True, "helion.autotune_effort": "none"})
        )

    def compile(self, fn, *args, tol=None):
        """Run compiled `fn`, check it against eager, and return the Helion kernel sources."""
        with self.assertNoLogs("torch._inductor.scheduler", level="CRITICAL"):
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        self.assertEqual(result, fn(*args), atol=tol, rtol=tol)
        self.code = "\n".join(codes)
        return SOURCE_RE.findall(self.code)

    def test_reduction_source(self):
        (source,) = self.compile(
            lambda x, b: (x * 2).sum(-1) + b,
            torch.randn(64, 100, device=GPU_TYPE),
            torch.randn(64, device=GPU_TYPE),
            tol=1e-4,
        )
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_x in hl.tile(64):
        acc = hl.zeros([tile_x], dtype=torch.float32)
        for tile_r in hl.tile(100):
            tmp0 = in0[tile_x, tile_r]
            tmp1 = hl.full([], 2.0, dtype=torch.float32)
            tmp2 = tmp0 * tmp1
            acc = acc + torch.sum(tmp2, -1)
        tmp3 = in1[tile_x]
        tmp4 = acc + tmp3
        out0[tile_x] = tmp4
""",
        )

    def test_pointwise(self):
        (source,) = self.compile(
            lambda x: torch.relu(x) * 2 + 1, torch.randn(64, 100, device=GPU_TYPE)
        )
        self.assertIn("for tile_x in hl.tile(6400):", source)

    def test_inplace_reuse_chain(self):
        def fn(x, w):
            y = torch.nn.functional.conv2d(x, w, padding=1)
            for i in range(32):
                y = y * 0.999 + 0.001 * (i + 1)
            return y + y.cos()

        x = torch.randn(1, 1, 64, 128, device=GPU_TYPE)
        w = torch.randn(1, 1, 3, 3, device=GPU_TYPE)
        (source,) = self.compile(fn, x, w, tol=1e-4)
        # The conv output buf0 becomes buf1, which becomes buf2, so buf0 is passed as buf2.
        self.assertIn("buf2 = buf1; del buf1  # reuse", self.code)
        self.assertRegex(self.code, r"helion_\w+\(reinterpret_tensor\(buf2, ")

    @parametrize("case", ["softmax", "layer_norm", "argmax", "dynamic"])
    def test_triton_fallback(self, case):
        x = torch.randn(64, 100, device=GPU_TYPE)
        fn = {
            "softmax": lambda x: torch.softmax(x, -1),
            "layer_norm": lambda x: torch.nn.functional.layer_norm(x, (100,)),
            "argmax": lambda x: x.argmax(-1),
            "dynamic": lambda x: x.sum(-1),
        }[case]
        if case == "dynamic":
            torch._dynamo.mark_dynamic(x, 0)
        metrics.reset()
        self.assertEqual(self.compile(fn, x, tol=1e-4), [])
        self.assertEqual(metrics.generated_kernel_count, self.code.count("@triton.jit"))


if __name__ == "__main__":
    run_tests()
