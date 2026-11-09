# Owner(s): ["module: inductor"]
from __future__ import annotations

import contextlib
import importlib.util
import re
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sympy

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codegen.helion import (
    HelionOverrides,
    HelionPrinter,
    HelionScheduling,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_HELION


HAS_CUTE = importlib.util.find_spec("cutlass") is not None
SOURCE_RE = re.compile(r'helion_\w+_source = """\\\n(.*?)"""', re.DOTALL)
BACKENDS = ["triton", "cute"]


def randn(*shape, dtype=torch.bfloat16):
    return torch.randn(shape, device="cuda", dtype=dtype)


@contextlib.contextmanager
def fp32_precision(precision):
    prior = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = precision
    try:
        yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = prior


@unittest.skipUnless(
    HAS_CUDA_AND_TRITON and HAS_HELION, "requires CUDA, Triton and Helion"
)
@instantiate_parametrized_tests
class TestHelionNativeMatmul(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        self.enterContext(config.patch({"helion.native_matmul": True}))

    def compile(
        self, fn, *args, expected=None, tol=None, backend="triton", helion=True
    ):
        """Run compiled `fn`, check it, and return the Helion kernel modules.

        By default the result is checked against `fn` on FP32 copies of
        FP16/BF16 inputs, since Helion kernels keep their epilogue in FP32.
        """
        if backend == "cute" and not HAS_CUTE:
            self.skipTest("requires CuTeDSL")
        with config.patch({"helion.backend": backend}):
            result, codes = run_and_get_code(torch.compile(fn, fullgraph=True), *args)
        if expected is None:
            low = (torch.float16, torch.bfloat16)
            expected = fn(*(a.float() if a.dtype in low else a for a in args))
            expected = pytree.tree_map(lambda e, r: e.to(r.dtype), expected, result)
        self.assertEqual(result, expected, atol=tol, rtol=tol)
        self.code = "\n".join(codes)
        sources = SOURCE_RE.findall(self.code)
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

@helion.kernel(backend="triton", static_shapes=True, autotune_effort="none", dot_precision="ieee")
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
    @parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_mm(self, dtype, backend):
        (source,) = self.compile(
            lambda a, b: a @ b,
            randn(40, 72, dtype=dtype),
            randn(72, 136, dtype=dtype),
            backend=backend,
        )
        self.assertIn(f'backend="{backend}"', source)
        # Operands keep their storage dtype (see HelionKernel.load_dot_operand).
        self.assertIn("tmp0 = in0[tile_y, tile_r]\n", source)
        self.assertIn("tmp1 = in1[tile_r, tile_x]\n", source)
        store = "out0[tile_y, tile_x] = acc"
        self.assertIn(
            store if dtype == torch.float32 else f"{store}.to({dtype})", source
        )

    def test_fp32_precision(self):
        with fp32_precision("tf32"):
            (source,) = self.compile(
                lambda a, b: a @ b,
                randn(40, 72, dtype=torch.float32),
                randn(72, 136, dtype=torch.float32),
                tol=5e-2,
            )
        self.assertIn('dot_precision="tf32"', source)

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

    @parametrize("backend", BACKENDS)
    @parametrize("batched", [False, True])
    @parametrize("bias", ["row", "column", "matrix", "scalar"])
    def test_epilogue_broadcast(self, bias, batched, backend):
        batch = (3,) if batched else ()
        shape = {"row": (40, 1), "column": (136,), "matrix": (40, 136), "scalar": ()}
        (source,) = self.compile(
            lambda a, b, c: torch.matmul(a, b) + c,
            randn(*batch, 40, 72),
            randn(*batch, 72, 136),
            randn(*shape[bias]),
            backend=backend,
        )
        load = {
            "row": "in2[tile_y, 0]",
            "column": "in2[tile_x]",
            "matrix": "in2[tile_y, tile_x]",
            "scalar": "in2[()]",
        }[bias]
        self.assertIn(f" = {load}.to(torch.float32)\n", source)
        # Values broadcast against the tile; only a row vector needs a reshape.
        if bias == "row":
            self.assertIn(".reshape([tile_y, 1])", source)
        else:
            self.assertNotIn(".reshape(", source)

    def test_bmm_epilogue_broadcast(self):
        (source,) = self.compile(
            lambda a, b, row, col: torch.bmm(a, b) + row + col,
            randn(3, 40, 72),
            randn(3, 72, 136),
            randn(3, 40, 1),
            randn(3, 1, 136),
        )
        self.assertIn("in2[tile_z, tile_y, 0].to(torch.float32)", source)
        self.assertIn(".reshape([tile_z, tile_y, 1])", source)
        self.assertIn("in3[tile_z, 0, tile_x].to(torch.float32)", source)
        self.assertIn(".reshape([tile_z, 1, tile_x])", source)

    @parametrize(
        "op",
        [
            "relu",
            "abs",
            "exp",
            "exp2",
            "expm1",
            "log",
            "log1p",
            "log2",
            "sqrt",
            "rsqrt",
            "sin",
            "cos",
            "tanh",
            "erf",
            "sigmoid",
            "floor",
            "ceil",
            "reciprocal",
        ],  # fmt: skip
    )
    def test_unary_epilogue(self, op):
        fn = getattr(torch, op)
        (source,) = self.compile(
            lambda a, b, c: a @ b + fn(c),
            randn(40, 72),
            randn(72, 136),
            torch.rand(136, device="cuda") + 0.5,
        )
        self.assertIn(f"torch.{op}(", source)

    def test_select_and_logical_epilogue(self):
        def fn(a, b, c):
            result = a @ b
            mask = torch.logical_or(torch.logical_and(result > 0, c < 1), ~(c > 2))
            return torch.where(mask, torch.minimum(result, c), torch.maximum(result, c))

        (source,) = self.compile(fn, randn(40, 72), randn(72, 136), randn(136))
        for op in ("where", "minimum", "maximum", "logical_and", "logical_or"):
            self.assertIn(f"torch.{op}(", source)

    @parametrize("backend", BACKENDS)
    def test_fma_epilogue(self, backend):
        # addcmul lowers to ops.fma, which rounds x * y + z once.
        (source,) = self.compile(
            lambda a, b, c, d: torch.addcmul(c, a @ b, d),
            randn(40, 72),
            randn(72, 136),
            randn(136),
            randn(40, 136),
            backend=backend,
        )
        self.assertIn(" = torch.addcmul(tmp2, acc, tmp3)", source)

    @parametrize("backend", BACKENDS)
    def test_masked_epilogue_load(self, backend):
        (source,) = self.compile(
            lambda a, b, first, second: a @ b + torch.cat((first, second)),
            randn(40, 72),
            randn(72, 136),
            randn(64),
            randn(72),
            backend=backend,
        )
        self.assertIn("hl.load(in2, [tile_x.index], extra_mask=", source)
        self.assertIn("hl.load(in3, [(-64) + tile_x.index], extra_mask=", source)

    def test_masked_epilogue_load_over_other_axes(self):
        # The row mask of the cat cannot change the address of bias[x], so the
        # bias load is unmasked and only torch.where selects its rows.
        (source,) = self.compile(
            lambda a, b, bias, rest: a @ b + torch.cat((bias.expand(10, 136), rest)),
            randn(40, 72),
            randn(72, 136),
            randn(136),
            randn(30, 136),
        )
        self.assertIn(" = in2[tile_x].to(torch.float32)", source)
        self.assertIn("hl.load(in3, [", source)

    @parametrize("batched", [False, True])
    def test_index_expr_epilogue(self, batched):
        batch = (3,) if batched else ()

        def fn(a, b):
            numel = 3 * 40 * 136 if batched else 40 * 136
            index = torch.arange(numel, device=a.device).reshape(*batch, 40, 136)
            return torch.matmul(a, b) + index * 1e-3

        (source,) = self.compile(fn, randn(*batch, 40, 72), randn(*batch, 72, 136))
        self.assertIn("136*tile_y.index[:, None]", source)
        self.assertIn("tile_x.index", source)
        if batched:
            self.assertIn("5440*tile_z.index[:, None, None]", source)

    def test_index_expr_wider_than_int32(self):
        def fn(a, b):
            index = torch.arange(40 * 136, device=a.device).reshape(40, 136)
            return torch.where(index * 10**8 % 7 == 0, a @ b, 0.0)

        (source,) = self.compile(fn, randn(40, 72), randn(72, 136))
        self.assertIn("tile_x.index.to(torch.int64)", source)

    @parametrize("case", ["reshape", "offset_transpose", "same_storage"])
    def test_operand_views(self, case):
        if case == "reshape":
            fn = lambda a, b: a.reshape(33, 72) @ b  # noqa: E731
            args = [randn(33, 9, 8), randn(72, 136)]
            view = "reinterpret_tensor(arg0_1, (33, 72), (72, 1), 0)"
        elif case == "offset_transpose":
            fn = lambda a, bt: a[1:].reshape(33, 72) @ bt.T  # noqa: E731
            args = [randn(34, 9, 8), randn(136, 72)]
            view = "reinterpret_tensor(arg1_1, (72, 136), (1, 72), 0)"
        else:
            fn, args = lambda a: a @ a.T, [randn(40, 72)]
            view = "reinterpret_tensor(arg0_1, (72, 40), (1, 72), 0)"
        (source,) = self.compile(fn, *args)
        # The wrapper passes 2D views of the inputs, so the kernel never decodes
        # a flat index with // or %.
        self.assertIn(view, self.code)
        self.assertIn("in0[tile_y, tile_r]", source)
        self.assertIn("in1[tile_r, tile_x]", source)
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)

    def test_operand_shared_with_epilogue(self):
        # x is both the lhs and the epilogue input; the same view of it is one
        # kernel parameter.
        (source,) = self.compile(lambda x, w: x @ w + x, randn(40, 72), randn(72, 72))
        self.assertIn("in0[tile_y, tile_r]", source)
        self.assertIn("in0[tile_y, tile_x]", source)
        self.assertNotIn("in2", source)

    @parametrize("batched", [False, True])
    @parametrize("layout", ["contiguous", "strided", "permuted"])
    @config.patch(comprehensive_padding=False)
    def test_epilogue_views(self, layout, batched):
        batch = (3,) if batched else ()
        matrix = {
            "contiguous": lambda: randn(*batch, 5, 8, 136),
            "strided": lambda: randn(*batch, 5, 8, 144)[..., :136],
            # An (8, 5, 136) input that fn transposes back to a contiguous layout.
            "permuted": lambda: randn(*batch, 5, 8, 136).transpose(-3, -2),
        }[layout]()

        def fn(a, b, matrix, row):
            if layout == "permuted":
                matrix = matrix.transpose(-3, -2)
            return (torch.matmul(a, b).reshape(*batch, 5, 8, 136) + matrix) * row

        (source,) = self.compile(
            fn,
            randn(*batch, 40, 72),
            randn(*batch, 72, 136),
            matrix,
            randn(*batch, 5, 8, 1),
        )
        axes = "tile_z, tile_y, tile_x" if batched else "tile_y, tile_x"
        self.assertIn(f"in2[{axes}]", source)
        self.assertIn(f"out0[{axes}] = ", source)
        self.assertNotIn(".index", source)

    @parametrize("case", ["transposed", "padded"])
    def test_flat_epilogue_input(self, case):
        # A transposed or padded [5, 8, 136] input is no 2D view of its storage,
        # so the kernel reads a flat view at the flat index, like Triton.
        if case == "transposed":
            bias, numel = randn(5, 136, 8).transpose(1, 2), 5440
        else:
            bias, numel = randn(5, 9, 150)[:, :8, :136], 6586
        (source,) = self.compile(
            lambda a, b, bias: (a @ b).reshape(5, 8, 136) + bias,
            randn(40, 72),
            randn(72, 136),
            bias,
        )
        self.assertIn(f"reinterpret_tensor(arg2_1, ({numel}, ), (1, ), 0)", self.code)
        self.assertRegex(source, r"in2\[.*tile_y\.index\[:, None\] // 8.*\]")

    def test_flat_epilogue_input_skipping_an_axis(self):
        # The [B, 1, 1, N] bias of a [B * H, M, N] bmm is read at
        # N * (z // H) + x, a flat index over the z and x tiles only.
        (source,) = self.compile(
            lambda a, b, bias: torch.bmm(a, b).view(2, 3, 40, 136) + bias,
            randn(6, 40, 72),
            randn(6, 72, 136),
            randn(2, 1, 1, 136),
        )
        self.assertIn(
            "in2[tile_x.index + 136*(tile_z.index[:, None, None] // 3)]", source
        )

    @parametrize("case", ["compute", "repeat", "gather", "permute_copy"])
    def test_materialized_operand(self, case):
        if case == "compute":
            fn, args = lambda a, b: (a + 1) @ (b - 2), [randn(40, 72), randn(72, 136)]
        elif case == "repeat":
            fn, args = lambda a, b: a.repeat(1, 9) @ b, [randn(40, 8), randn(72, 136)]
        elif case == "gather":
            indices = torch.randint(0, 80, (40,), device="cuda")
            fn, args = (
                lambda a, i, b: a[i] @ b,
                [randn(80, 72), indices, randn(72, 136)],
            )
        else:
            fn = lambda a, b: a.permute(1, 0, 2).reshape(40, 64) @ b  # noqa: E731
            args = [randn(8, 40, 8), randn(64, 136)]
        (source,) = self.compile(fn, *args)
        # The producer stays a Triton kernel and the Helion kernel reads its
        # output buffer, not the input.
        self.assertIn("@triton.jit", self.code)
        self.assertRegex(self.code, r"helion_\w+\((reinterpret_tensor\()?buf0, ")
        self.assertIn("in0[tile_y, tile_r]", source)

    @parametrize("backend", BACKENDS)
    @parametrize("epilogue", [False, True])
    def test_horizontal(self, epilogue, backend):
        def fn(a, b, c):
            first, second = a @ b, a @ c
            return (
                torch.relu(first - second + first * second)
                if epilogue
                else (first, second)
            )

        (source,) = self.compile(
            fn, randn(40, 72), randn(72, 136), randn(72, 136), backend=backend
        )
        self.assertIn("acc1 = hl.zeros(", source)
        self.assertIn("acc1 = torch.addmm(acc1, tmp0, ", source)
        # The shared lhs is loaded once, and fused intermediates are not stored.
        self.assertEqual(source.count("in0[tile_y, tile_r]"), 1)
        self.assertIn("in2[tile_r, tile_x]", source)
        self.assertEqual("out1[" in source, not epilogue)

    def test_horizontal_dots_share_one_k_loop(self):
        # The nodes fuse as [dot1, cast1, dot2, cast2]; both dots still go in
        # one K loop, which loads w once.
        (source,) = self.compile(
            lambda w, a, b: ((a @ w).float(), (b @ w).float()),
            randn(16, 512),
            randn(512, 16),
            randn(512, 16),
        )
        self.assertEqual(source.count("for tile_r in hl.tile(16):"), 1)
        self.assertEqual(source.count(" = in1[tile_r, tile_x]"), 1)
        self.assertIn("acc1 = torch.addmm(acc1, tmp2, tmp1)", source)

    def test_horizontal_operand_views(self):
        (source,) = self.compile(
            lambda a, b: a[:-1] @ b + a[1:] @ b, randn(41, 72), randn(72, 136)
        )
        self.assertIn("in2[tile_y, tile_r]", source)
        self.assertIn("reinterpret_tensor(arg0_1, (40, 72), (72, 1), 72)", self.code)

    def test_identical_kernels_share_a_module(self):
        def fn(a, b):
            return torch.relu(torch.relu(a @ b) @ b)

        args = (randn(32, 64), randn(64, 64))
        # Like eager, the first product is rounded to bf16 before the second.
        self.compile(fn, *args, expected=fn(*args))
        (name,) = re.findall(r"(helion_\w+)_source = ", self.code)
        self.assertEqual(self.code.count(f"{name}("), 2)

    @parametrize("producer", ["cumsum", "sort"])
    def test_epilogue_input_reused_for_output(self, producer):
        # x's producer is a separate Triton kernel and the epilogue is the last
        # use of its output, so the output takes over that buffer
        # (buf1 = buf0; del buf0) and the Helion kernel reads it through buf1.
        def fn(a, b, x):
            y = torch.cumsum(x, -1) if producer == "cumsum" else torch.sort(x, -1)[0]
            return a @ b + y

        self.compile(fn, randn(40, 72), randn(72, 136), randn(40, 136), tol=5e-2)
        self.assertIn("# reuse", self.code)

    def test_mutation_epilogue(self):
        a, b, x = randn(40, 72), randn(72, 136), randn(40, 136)
        expected = (x.float() + a.float() @ b.float()).to(x.dtype)
        (source,) = self.compile(
            lambda a, b, x: x.add_(a @ b), a, b, x, expected=expected
        )
        self.assertIn("in2[tile_y, tile_x] = ", source)
        self.assertEqual(x, expected)

    def test_output_alias(self):
        def fn(a):
            result = a @ a.T
            return result, result.T, torch.relu(result)

        (source,) = self.compile(fn, randn(32, 64))
        self.assertIn("torch.relu(", source)

    @parametrize(
        "case",
        [
            "m_is_one",
            "n_is_one",
            "k_is_one",
            "self_product_of_a_row",
            "float64",
            "dynamic",
            "indirect_epilogue",
            "cpu_scalar",
        ],
    )
    def test_fallback(self, case):
        a, b = randn(32, 64), randn(64, 48)
        fn = lambda a, b: torch.relu(a @ b)  # noqa: E731
        args = [a, b]
        if case == "m_is_one":
            args = [randn(1, 64), b]
        elif case == "n_is_one":
            args = [a, randn(64, 1)]
        elif case == "k_is_one":
            args = [randn(32, 1), randn(1, 48)]
        elif case == "self_product_of_a_row":
            fn, args = lambda x: x @ x.T, [randn(1, 64)]
        elif case == "float64":
            args = [a.double(), b.double()]
        elif case == "dynamic":
            torch._dynamo.mark_dynamic(a, 0)
        elif case == "indirect_epilogue":
            fn = lambda a, b, c, i: a @ b + c[i]  # noqa: E731
            args = [a, b, randn(80, 48), torch.randint(0, 80, (32,), device="cuda")]
        else:
            fn = lambda a, b, c: a @ b + c  # noqa: E731
            args = [a, b, torch.tensor(2.0)]
        # Unfused epilogues round the product first, as eager does.
        sources = self.compile(fn, *args, expected=fn(*args), tol=2e-2, helion=False)
        if case in ("indirect_epilogue", "cpu_scalar"):
            # The matmul is a Helion kernel and the epilogue a Triton kernel.
            self.assertEqual(len(sources), 1)
            self.assertIn("@triton.jit", self.code)
        else:
            self.assertEqual(sources, [])

    def test_addmm(self):
        (source,) = self.compile(
            lambda bias, a, b: torch.addmm(bias, a, b),
            randn(136),
            randn(40, 72),
            randn(72, 136),
        )
        # Unit alpha and beta add the bias without scaling it.
        self.assertIn("tmp3 = tmp2 + acc", source)

    def test_baddbmm(self):
        (source,) = self.compile(
            lambda bias, a, b: torch.baddbmm(bias, a, b),
            randn(3, 40, 136),
            randn(3, 40, 72),
            randn(3, 72, 136),
        )
        self.assertIn("tmp3 = tmp2 + acc", source)

    def test_addmm_out_dtype(self):
        (source,) = self.compile(
            lambda bias, a, b: torch.addmm(
                bias, a, b, out_dtype=torch.float32, alpha=1.25, beta=0.75
            ),
            randn(136, dtype=torch.float32),
            randn(40, 72),
            randn(72, 136),
        )
        self.assertIn("hl.full([], 1.25, dtype=torch.float32)", source)

    @config.patch(max_autotune=True)
    def test_mm_plus_mm(self):
        (source,) = self.compile(
            lambda a, b, c, d: a @ b + c @ d,
            randn(32, 64),
            randn(64, 48),
            randn(32, 64),
            randn(64, 48),
        )
        self.assertEqual(source.count(" = torch.addmm("), 2)

    def test_mixed_wrapper(self):
        def fn(a, b):
            result = a @ b
            return torch.sinh(result), result.sum(dim=-1)

        args = (randn(32, 72), randn(72, 64))
        (source,) = self.compile(fn, *args, expected=fn(*args), tol=2e-2)
        # sinh is not a Helion op, and a row sum is not fused into a Helion
        # kernel, so both run as Triton kernels after it.
        self.assertIn("@triton.jit", self.code)
        self.assertNotIn("torch.sinh(", source)
        self.assertNotIn("sum", source)

    def test_triton_template_in_graph(self):
        from torch.nn.attention.flex_attention import flex_attention

        def fn(q, k, v, a, b):
            return torch.relu(flex_attention(q, k, v)), torch.relu(a @ b)

        q, k, v = (randn(1, 2, 128, 64) for _ in range(3))
        args = (q, k, v, randn(32, 64), randn(64, 48))
        # The flex_attention template (which uses tl.dot) and its relu epilogue
        # stay Triton.
        (source,) = self.compile(fn, *args, expected=fn(*args), tol=2e-2, helion=False)
        self.assertIn("torch.relu(", source)
        self.assertIn("triton_tem_fused", self.code)

    @parametrize("option", ["benchmark_fusion", "combo_kernels"])
    def test_optional_triton_paths(self, option):
        with config.patch({option: True}):
            (source,) = self.compile(
                lambda a, b: (torch.relu(a @ b), torch.sin(a), torch.cos(b)),
                randn(32, 64, dtype=torch.float32),
                randn(64, 48, dtype=torch.float32),
            )
        self.assertIn("torch.relu(", source)

    @parametrize("backend", BACKENDS)
    def test_cudagraph_tree(self, backend):
        def fn(a, b):
            return torch.relu((a + 1) @ b), a.sum(dim=-1)

        if backend == "cute" and not HAS_CUTE:
            self.skipTest("requires CuTeDSL")
        args = [randn(32, 64), randn(64, 48)]
        counters.clear()
        with config.patch({"triton.cudagraphs": True, "helion.backend": backend}):
            compiled = torch.compile(fn, fullgraph=True)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*args), fn(*args))
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    @parametrize("backend", BACKENDS)
    def test_training_cudagraph(self, backend):
        def fn(a, b):
            return torch.relu((a + 1) @ b).sum()

        if backend == "cute" and not HAS_CUTE:
            self.skipTest("requires CuTeDSL")
        a = randn(32, 64, dtype=torch.float32).requires_grad_()
        b = randn(64, 48, dtype=torch.float32).requires_grad_()
        counters.clear()
        with config.patch({"triton.cudagraphs": True, "helion.backend": backend}):
            compiled = torch.compile(fn, fullgraph=True)
            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                a.grad = b.grad = None
                compiled(a, b).backward()
                ra, rb = a.detach().requires_grad_(), b.detach().requires_grad_()
                fn(ra, rb).backward()
                self.assertEqual(a.grad, ra.grad, atol=1e-4, rtol=1e-4)
                self.assertEqual(b.grad, rb.grad, atol=1e-4, rtol=1e-4)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    @parametrize("effort", ["none", "quick", "full"])
    def test_kernel_settings(self, effort):
        # Only the decorator is checked, so skip running (and autotuning) it.
        import helion

        with (
            config.patch({"helion.autotune_effort": effort}),
            patch.object(helion.Kernel, "__call__", lambda self, *args: None),
        ):
            _, (code,) = run_and_get_code(
                torch.compile(lambda a, b: a @ b), randn(32, 64), randn(64, 48)
            )
        self.assertIn(
            '@helion.kernel(backend="triton", static_shapes=True, '
            f'autotune_effort="{effort}", dot_precision="ieee")',
            code,
        )


@instantiate_parametrized_tests
class TestHelionCodegenUnits(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @parametrize(
        "source",
        [
            "import torch\ndef kernel():\n    return 1\n",
            'value = """triple quoted"""\n',
            "path = 'a\\\\b'\nvalue = '\\n'\n",
            "",
        ],
    )
    def test_wrapper_source_roundtrip(self, source):
        cache = Mock()
        exec(
            HelionScheduling.format_kernel_definition("helion_0", source),
            {"PyCodeCache": cache},
        )
        cache.load.assert_called_once_with(source)

    def test_spelled_ops_match_torch(self):
        x = torch.rand(8) + 0.5
        for name, method in vars(HelionOverrides).items():
            spelled = getattr(method, "__func__", None)
            kernel_ops = (
                "constant",
                "to_dtype",
                "index_expr",
                "value_expr",
                "masked",
                "dot",
            )
            if spelled is None or name in kernel_ops:
                continue
            args = (
                [x > 1, x, x.flip(0)] if name == "where" else [x, x.flip(0), x.sqrt()]
            )
            args = args[: spelled.__code__.co_argcount]
            names = ["a", "b", "c"][: len(args)]
            result = eval(spelled(*names), {"torch": torch, **dict(zip(names, args))})
            if name == "fma":
                expected = args[0] * args[1] + args[2]
            else:
                expected = getattr(torch, name)(*args)
            self.assertEqual(result, expected, msg=name)

    @parametrize("op", [sympy.Min, sympy.Max])
    def test_printer_min_max(self, op):
        x = sympy.Symbol("x", integer=True)
        expression = HelionPrinter().doprint(op(x, 2**25 + 1) - x, simplify=False)
        values = torch.arange(2**25 - 1, 2**25 + 4)
        hl = SimpleNamespace(
            full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype)
        )
        result = eval(expression, {"torch": torch, "hl": hl, "x": values})
        reference = (torch.minimum if op is sympy.Min else torch.maximum)(
            values, torch.tensor(2**25 + 1)
        ) - values
        self.assertEqual(result, reference)


if __name__ == "__main__":
    run_tests()
