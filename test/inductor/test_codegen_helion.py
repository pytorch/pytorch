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
from torch._inductor.codegen.cuda_combined_scheduling import CUDACombinedScheduling
from torch._inductor.codegen.helion import helion_scheduling
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification, parametrize
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


def randn(device, *shape, dtype=torch.bfloat16):
    return torch.randn(shape, device=device, dtype=dtype)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
class TestHelionNativeMatmul(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        self.enterContext(
            config.patch({"helion.enabled": True, "helion.autotune_effort": "none"})
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

    def test_mm_source(self, device):
        (source,) = self.compile(
            lambda a, b, bias: torch.relu(a @ b + bias),
            randn(device, 40, 72),
            randn(device, 72, 136),
            randn(device, 136),
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
    def test_bmm_epilogue(self, device, backend):
        (source,) = self.compile(
            lambda a, b, bias: torch.relu(torch.bmm(a, b)) + bias,
            randn(device, 3, 40, 72),
            randn(device, 3, 72, 136),
            randn(device, 136),
            backend=backend,
        )
        self.assertIn("for tile_z, tile_y, tile_x in hl.tile([3, 40, 136]):", source)
        self.assertIn("tmp0 = in0[tile_z, tile_y, tile_r]", source)
        self.assertIn("tmp1 = in1[tile_z, tile_r, tile_x]", source)
        self.assertIn("acc = torch.baddbmm(acc, tmp0, tmp1)", source)
        self.assertIn("out0[tile_z, tile_y, tile_x] = ", source)

    def test_computed_operand(self, device):
        a, b = randn(device, 40, 72), randn(device, 72, 136)
        (source,) = self.compile(lambda a, b: (a + 1) @ (b - 2), a, b)
        self.assertIn("in0[tile_y, tile_r]", source)
        self.assertIn("in1[tile_r, tile_x]", source)

    def test_saved_bf16_output(self, device):
        # silu's backward reads a @ b, so the bf16 product is saved and rounded.
        fn = lambda a, b: torch.nn.functional.silu(a @ b)  # noqa: E731
        a, b = randn(device, 40, 72), randn(device, 72, 136).requires_grad_()
        (source,) = self.compile(fn, a, b, expected=fn(a, b), tol=2e-2)
        self.assertExpectedInline(
            source,
            """\
import torch
import helion
import helion.language as hl

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none")
def kernel(in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([40, 136]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(72):
            tmp0 = in0[tile_y, tile_r]
            tmp1 = in1[tile_r, tile_x]
            acc = torch.addmm(acc, tmp0, tmp1)
        tmp2 = acc.to(torch.bfloat16)
        tmp3 = tmp2.to(torch.float32)
        out0[tile_y, tile_x] = tmp3.to(torch.bfloat16)
        tmp4 = -tmp3
        tmp5 = torch.exp(tmp4)
        tmp6 = hl.full([], 1.0, dtype=torch.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tmp3 / tmp7
        out1[tile_y, tile_x] = tmp8.to(torch.bfloat16)
""",
        )

    def test_cast_epilogue(self, device):
        # Integers, so that the int32 cast is exact.
        a = torch.randint(-3, 4, (40, 72), device=device, dtype=torch.float32)
        b = torch.randint(-3, 4, (72, 136), device=device, dtype=torch.float32)
        (source,) = self.compile(lambda a, b: (a @ b).to(torch.int32) + 1, a, b)
        self.assertIn("tmp2 = acc.to(torch.int32)", source)

    def test_upcast_prologue(self, device):
        # a + 1 is computed in fp32, so the bf16 a is read as fp32.
        fn = lambda a, b: (a + 1).float() @ b.float()  # noqa: E731
        a, b = randn(device, 40, 72), randn(device, 72, 136)
        (source,) = self.compile(fn, a, b, tol=1e-3)
        self.assertIn("in0[tile_y, tile_r].to(torch.float32)", source)

    def test_out_dtype(self, device):
        fn = lambda a, b: torch.mm(a, b, out_dtype=torch.float32)  # noqa: E731
        a, b = randn(device, 40, 72), randn(device, 72, 136)
        (source,) = self.compile(fn, a, b, tol=1e-3)
        self.assertIn("out0[tile_y, tile_x] = acc\n", source)

    def test_operand_view_of_dynamic_input(self, device):
        x = randn(device, 4, 40, 72)
        torch._dynamo.mark_dynamic(x, 0)
        # x[0] is a static [40, 72] view of the [s, 40, 72] input.
        (source,) = self.compile(lambda x, w: x[0] @ w, x, randn(device, 72, 136))
        self.assertIn("in0[tile_y, tile_r]", source)
        self.assertRegex(
            self.code, r"reinterpret_tensor\(arg\d+_1, \(40, 72\), \(72, 1\), 0\)"
        )

    @config.patch(nan_asserts=True)
    def test_nan_asserts_check_the_tensor(self, device):
        base = randn(device, 5, 9, 150)
        base[:, 8:] = float("nan")
        # bias holds no NaN, but its flat operand view spans base's NaN rows.
        bias = base[:, :8, :136]
        self.compile(
            lambda a, b, bias: (a @ b).reshape(5, 8, 136) + bias,
            randn(device, 40, 72),
            randn(device, 72, 136),
            bias,
        )
        self.assertIn("reinterpret_tensor(arg2_1, (6586, ), (1, ), 0)", self.code)
        self.assertIn("assert not arg2_1.isnan().any().item()", self.code)

    @parametrize("epilogue", [False, True])
    def test_horizontal(self, device, epilogue):
        def fn(a, b, c):
            first, second = a @ b, a @ c
            return (
                torch.relu(first - second + first * second)
                if epilogue
                else (first, second)
            )

        args = randn(device, 40, 72), randn(device, 72, 136), randn(device, 72, 136)
        (source,) = self.compile(fn, *args)
        self.assertIn("acc1 = hl.zeros(", source)
        self.assertIn("acc1 = torch.addmm(acc1, tmp0, ", source)
        self.assertEqual(source.count("in0[tile_y, tile_r]"), 1)
        self.assertIn("in2[tile_r, tile_x]", source)
        self.assertEqual("out1[" in source, not epilogue)

    def test_masked_epilogue_load(self, device):
        (source,) = self.compile(
            lambda a, b, first, second: a @ b + torch.cat((first, second)),
            randn(device, 40, 72),
            randn(device, 72, 136),
            randn(device, 64),
            randn(device, 72),
        )
        self.assertIn("hl.load(in2, [tile_x.index], extra_mask=", source)
        self.assertIn("hl.load(in3, [(-64) + tile_x.index], extra_mask=", source)

    @parametrize("case", ["horizontal", "vertical", "y_only", "x_only"])
    def test_reduction_over_dot_loop(self, device, case):
        def horizontal(a, b):
            return a @ b, (a[:, :, None] * b[None]).sum(1)

        def vertical(a, b):
            # The sum reads the matmul output in a second K loop.
            c = a @ b
            return c, (c[:, :, None] * b[:, 0]).sum(-1)

        def y_only(a, b):
            return a @ b, a[:, :, None].expand(40, 72, 136).sum(1)

        def x_only(a, b):
            return a @ b, b[None].expand(40, 72, 136).sum(1)

        # A sum over the dot's ranges reduces tile_r where the axis order puts it.
        fn, line = {
            "horizontal": (horizontal, r"acc1 = acc1 \+ torch\.sum\(tmp\d+, -2\)\n"),
            "vertical": (vertical, r"acc1 = acc1 \+ torch\.sum\(tmp\d+, -2\)\n"),
            "y_only": (y_only, r"torch\.sum\(tmp\d+, -1\)\.reshape\(\[tile_y, 1\]\)"),
            "x_only": (x_only, r"acc1 = acc1 \+ torch\.sum\(tmp\d+, -2\)\n"),
        }[case]
        (source,) = self.compile(fn, randn(device, 40, 72), randn(device, 72, 136))
        self.assertRegex(source, line)

    @parametrize("case", ["pad", "tril", "constant", "vector", "permuted"])
    def test_operand(self, device, case):
        b = randn(device, 72, 136)
        fn, a, lines = {
            # The mask needs tile_y and tile_r, and a [40, 72] view of the [36, 64]
            # input would leave its storage, so the load is flat.
            "pad": (
                lambda a, b: torch.nn.functional.pad(a, (0, 8, 0, 4)) @ b,
                randn(device, 36, 64),
                [".reshape([tile_y, 1])", "extra_mask="],
            ),
            "tril": (
                lambda a, b: a.tril() @ b,
                randn(device, 40, 72),
                ["torch.where("],
            ),
            "constant": (
                lambda a, b: a @ torch.ones_like(b),
                randn(device, 40, 72),
                [".expand(tile_r, tile_x)"],
            ),
            "vector": (
                lambda a, b: a @ b[:, :1].expand(72, 136),
                randn(device, 40, 72),
                [".reshape([tile_r, 1]).expand(tile_r, tile_x)"],
            ),
            "permuted": (
                lambda a, b: a.permute(1, 0, 2).reshape(40, 72) @ b,
                randn(device, 8, 40, 9),
                ["//"],
            ),
        }[case]
        (source,) = self.compile(fn, a, b)
        for line in lines:
            self.assertIn(line, source)

    @parametrize(
        "case",
        [
            "dynamic",
            "gathered_operand",
            "cpu_scalar",
            "native_matmul_off",
            "persistent_tma",
        ],
    )
    def test_fallback(self, device, case):
        a, b = randn(device, 32, 64), randn(device, 64, 48)
        fn = lambda a, b: torch.relu(a @ b)  # noqa: E731
        args = [a, b]
        if case == "dynamic":
            torch._dynamo.mark_dynamic(a, 0)
        elif case == "gathered_operand":
            fn = lambda a, i, b: a[i] @ b  # noqa: E731
            i = torch.randint(0, 80, (32,), device=device)
            args = [randn(device, 80, 64), i, b]
        elif case == "cpu_scalar":
            # The CPU scalar epilogue fuses into the matmul, so the kernel is Triton.
            fn = lambda a, b, c: a @ b + c  # noqa: E731
            args = [a, b, torch.tensor(2.0, device="cpu")]
        elif case == "native_matmul_off":
            self.enterContext(config.patch({"helion.native_matmul": False}))
        else:
            # Triton's native matmul codegen, which emits what Helion declines,
            # cannot use TMA.
            patch = {"triton.enable_persistent_tma_matmul": True}
            self.enterContext(config.patch(patch))
        sources = self.compile(fn, *args, expected=fn(*args), tol=2e-2, helion=False)
        self.assertEqual(sources, [])
        if case == "native_matmul_off":
            self.assertIn("extern_kernels.mm(", self.code)

    def test_cudagraph_tree(self, device):
        def fn(a, b):
            return torch.relu((a + 1) @ b), a.sum(dim=-1)

        args = [randn(device, 32, 64), randn(device, 64, 48)]
        counters.clear()
        with config.patch({"triton.cudagraphs": True}):
            compiled = torch.compile(fn, fullgraph=True)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*args), fn(*args))
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)


instantiate_device_type_tests(TestHelionNativeMatmul, globals(), only_for=GPU_TYPE)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
class TestHelionDefaults(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_default_config(self):
        import helion

        args = randn(GPU_TYPE, 32, 64), randn(GPU_TYPE, 64, 48)
        with (
            config.patch({"helion.enabled": True}),
            patch.object(helion.Kernel, "__call__", lambda self, *args: None),
        ):
            _, (code,) = run_and_get_code(torch.compile(lambda a, b: a @ b), *args)
        (source,) = [s for s in SOURCE_RE.findall(code) if DOT_RE.search(s)]
        self.assertIn('autotune_effort="quick"', source)
        self.assertNotRegex(code, r"extern_kernels\.\w*mm\w*\(")

    @config.patch({"helion.enabled": True})
    def test_missing_helion(self):
        with (
            patch.object(helion_scheduling, "has_helion", lambda: False),
            self.assertRaisesRegex(RuntimeError, "Helion is not installed"),
        ):
            CUDACombinedScheduling(None)


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
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

    def test_reduction_source(self, device):
        (source,) = self.compile(
            lambda x, b: (x * 2).sum(-1) + b,
            torch.randn(64, 100, device=device),
            torch.randn(64, device=device),
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

    def test_pointwise(self, device):
        (source,) = self.compile(
            lambda x: torch.relu(x) * 2 + 1, torch.randn(64, 100, device=device)
        )
        self.assertIn("for tile_x in hl.tile(6400):", source)

    @parametrize("case", ["float_to_uint8", "int_to_fp8", "bool_to_fp8"])
    def test_cast(self, device, case):
        x = torch.randn(64, 100, device=device) * 100
        fn, line = {
            # eager wraps a negative value: (-5.3).to(torch.uint8) is 251.
            "float_to_uint8": (
                lambda x: (x * 2 - 1).to(torch.uint8),
                ".to(torch.int16).to(torch.uint8)",
            ),
            "int_to_fp8": (
                lambda x: (x.int() + 1).to(torch.float8_e4m3fn).float(),
                ".to(torch.float32).to(torch.float8_e4m3fn)",
            ),
            "bool_to_fp8": (
                lambda x: (x > 0).to(torch.float8_e5m2).float(),
                ".to(torch.float32).to(torch.float8_e5m2)",
            ),
        }[case]
        (source,) = self.compile(fn, x)
        self.assertIn(line, source)

    def test_inplace_reuse_chain(self, device):
        def fn(x, w):
            y = torch.nn.functional.conv2d(x, w, padding=1)
            for i in range(32):
                y = y * 0.999 + 0.001 * (i + 1)
            return y + y.cos()

        x = torch.randn(1, 1, 64, 128, device=device)
        w = torch.randn(1, 1, 3, 3, device=device)
        (source,) = self.compile(fn, x, w, tol=1e-4)
        # The conv output buf0 becomes buf1, which becomes buf2, so buf0 is passed as buf2.
        self.assertIn("buf2 = buf1; del buf1  # reuse", self.code)
        self.assertRegex(self.code, r"helion_\w+\(reinterpret_tensor\(buf2, ")

    @parametrize(
        "case",
        [
            "softmax",
            "layer_norm",
            "argmax",
            "dynamic",
            "strict_reduction",
            "strict_pointwise",
            "batch_invariant",
            "deterministic",
        ],
    )
    def test_triton_fallback(self, device, case):
        x = torch.randn(64, 100, device=device)
        fn = {
            "softmax": lambda x: torch.softmax(x, -1),
            "layer_norm": lambda x: torch.nn.functional.layer_norm(x, (100,)),
            "argmax": lambda x: x.argmax(-1),
        }.get(case, lambda x: x.sum(-1))
        if case == "dynamic":
            torch._dynamo.mark_dynamic(x, 0)
        # Without these modes, x.sum(-1) is a Helion kernel.
        modes = {
            "strict_reduction": {"numerics": "strict_reduction"},
            "strict_pointwise": {"numerics": "strict_pointwise"},
            "batch_invariant": {"batch_invariant": True},
            "deterministic": {"deterministic": True},
        }
        metrics.reset()
        with config.patch(modes.get(case, {})):
            self.assertEqual(self.compile(fn, x, tol=1e-4), [])
        self.assertEqual(metrics.generated_kernel_count, self.code.count("@triton.jit"))


instantiate_device_type_tests(TestHelionReduction, globals(), only_for=GPU_TYPE)


if __name__ == "__main__":
    run_tests()
