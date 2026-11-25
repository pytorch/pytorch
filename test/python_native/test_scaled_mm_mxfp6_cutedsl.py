# Owner(s): ["module: dsl-native-ops"]
#
# aten::_scaled_mm_v2 validates MXFP6 (float6_e2m3fn_x4) calls and runs them
# through aten::_scaled_mm_mxfp6, which the CuTeDSL override implements with the
# vendored QuACK block-scaled GEMM on SM100.

import math
import unittest

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_cuda import IS_SM10X, TEST_CUDA
from torch.testing._internal.common_quantized import (
    _float6_e2m3fn_x4_to_float32,
    to_blocked,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _override_registered() -> bool:
    from torch._native.registry import get_dsl_operations

    return "_scaled_mm_mxfp6.out" in get_dsl_operations("cutedsl")


def _quantize(value):
    """bf16 [R, K] -> (float6_e2m3fn_x4 [R, K/4], swizzled e8m0 scales, fp64 values)."""
    from torch._vendor.quack.blockscaled.quantize import to_mxfp6_e2m3_packed

    packed, scale = to_mxfp6_e2m3_packed(value, 32)
    data = packed.view(torch.float6_e2m3fn_x4)
    dequantized = _float6_e2m3fn_x4_to_float32(data).double()
    dequantized *= scale.double().repeat_interleave(32, dim=-1)
    return data, to_blocked(scale).flatten(), dequantized


def _operands(m, n, k):
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / math.sqrt(k)
    data_a, scale_a, ref_a = _quantize(a)
    data_b, scale_b, ref_b = _quantize(b)
    return data_a, data_b.t(), scale_a, scale_b, ref_a @ ref_b.t()


def _scaled_mm(a, b, scale_a, scale_b, **kwargs):
    kwargs.setdefault("swizzle_a", F.SwizzleType.SWIZZLE_32_4_4)
    kwargs.setdefault("swizzle_b", F.SwizzleType.SWIZZLE_32_4_4)
    return F.scaled_mm(
        a,
        b,
        scale_a,
        F.ScalingType.BlockWise1x32,
        scale_b,
        F.ScalingType.BlockWise1x32,
        **kwargs,
    )


def _scaled_mm_v2_args(a, b, scale_a, scale_b, out_dtype):
    block = [F.ScalingType.BlockWise1x32.value]
    swizzle = [F.SwizzleType.SWIZZLE_32_4_4.value]
    return (a, b, [scale_a], block, swizzle, [scale_b], block, swizzle, None, out_dtype)


@unittest.skipUnless(TEST_CUDA and IS_SM10X, "requires an SM100-family GPU")
@unittest.skipUnless(
    TEST_CUDA and _override_registered(), "CuTeDSL override not registered"
)
class TestScaledMmMXFP6CuTeDSL(TestCase):
    hw_classification = HardwareClassification.CUDA

    def assertMatches(self, out, reference, out_dtype=torch.bfloat16):
        self.assertEqual(out.dtype, out_dtype)
        tolerance = 1e-2 if out_dtype is torch.bfloat16 else 1e-4
        self.assertEqual(
            out.double(), reference, atol=tolerance * reference.abs().max(), rtol=0
        )

    # Shapes cover each branch of the config choice, M off the 128-row tile, and
    # an N whose bf16 rows are not 16-byte aligned.
    @parametrize(
        "m,n,k",
        [
            (16, 512, 256),
            (16, 13056, 256),
            (64, 13056, 256),
            (16, 26112, 128),
            (200, 520, 512),
            (640, 512, 1024),
            (16, 33, 128),
        ],
    )
    def test_matches_dequantized_reference(self, m, n, k):
        a, b, scale_a, scale_b, reference = _operands(m, n, k)
        self.assertMatches(_scaled_mm(a, b, scale_a, scale_b), reference)

    @parametrize("out_dtype", (torch.bfloat16, torch.float32))
    def test_out_dtype_and_out(self, out_dtype):
        a, b, scale_a, scale_b, reference = _operands(128, 256, 512)
        out = _scaled_mm(a, b, scale_a, scale_b, output_dtype=out_dtype)
        self.assertMatches(out, reference, out_dtype)
        args = _scaled_mm_v2_args(a, b, scale_a, scale_b, out_dtype)
        buffer = torch.empty(0, dtype=out_dtype, device="cuda")
        torch.ops.aten._scaled_mm_v2.out(*args, out=buffer)
        self.assertEqual(buffer, out, atol=0, rtol=0)
        strided = torch.empty(128, 512, dtype=out_dtype, device="cuda")[:, ::2]
        torch.ops.aten._scaled_mm_v2.out(*args, out=strided)
        self.assertEqual(strided, out, atol=0, rtol=0)

    def test_unaligned_out_and_scales(self):
        # Contiguous views that start off a 16-byte boundary.
        a, b, scale_a, scale_b, reference = _operands(128, 256, 512)
        scale_b = torch.empty(scale_b.numel() + 1, dtype=scale_b.dtype, device="cuda")[
            1:
        ].copy_(scale_b)
        out = torch.empty(128 * 256 + 1, dtype=torch.bfloat16, device="cuda")[1:]
        out = out.view(128, 256)
        args = _scaled_mm_v2_args(a, b, scale_a, scale_b, torch.bfloat16)
        torch.ops.aten._scaled_mm_v2.out(*args, out=out)
        self.assertMatches(out, reference)

    def test_mxfp6_op_out_and_fake(self):
        a, b, scale_a, scale_b, reference = _operands(128, 256, 512)
        out = torch.empty(7, dtype=torch.bfloat16, device="cuda")
        torch.ops.aten._scaled_mm_mxfp6.out(
            a, b, scale_a, scale_b, torch.bfloat16, out=out
        )
        self.assertMatches(out, reference)
        with self.assertRaisesRegex(ValueError, "expected out"):
            torch.ops.aten._scaled_mm_mxfp6.out(
                a, b, scale_a, scale_b, torch.bfloat16, out=out.cpu()
            )
        with FakeTensorMode() as mode:
            fake = mode.from_tensor
            result = torch.ops.aten._scaled_mm_mxfp6(
                fake(a), fake(b), fake(scale_a), fake(scale_b), torch.float32
            )
            self.assertEqual((result.shape, result.dtype), ((128, 256), torch.float32))

    def test_empty_and_zero_k(self):
        a, b, scale_a, scale_b, _ = _operands(128, 256, 512)
        self.assertEqual(_scaled_mm(a[:0], b, scale_a[:0], scale_b).shape, (0, 256))
        zero_k = _scaled_mm(a[:, :0], b[:0], scale_a[:0], scale_b[:0])
        self.assertEqual(
            zero_k, torch.zeros(128, 256, device="cuda", dtype=torch.bfloat16)
        )

    def test_out_with_unaligned_row_stride(self):
        # With M == 1 a row stride of 9 is still contiguous.
        a, b, scale_a, scale_b, reference = _operands(1, 8, 512)
        out = torch.empty_strided((1, 8), (9, 1), dtype=torch.bfloat16, device="cuda")
        args = _scaled_mm_v2_args(a, b, scale_a, scale_b, torch.bfloat16)
        torch.ops.aten._scaled_mm_v2.out(*args, out=out)
        self.assertMatches(out, reference)

    def test_mxfp6_op_validates_its_inputs(self):
        a, b, scale_a, scale_b, _ = _operands(128, 256, 512)
        op = torch.ops.aten._scaled_mm_mxfp6
        cases = {
            "on": (a, b, scale_a.cpu(), scale_b),
            "multiple of 128": (a[:, :16], b[:16], scale_a, scale_b),
            "float6_e2m3fn_x4": (a.view(torch.uint8), b, scale_a, scale_b),
            "elements": (a, b, scale_a[:-1], scale_b),
        }
        for message, args in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                op(*args, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "out_dtype"):
            op(a, b, scale_a, scale_b, torch.int32)
        # Scales are checked before ``out`` is touched, also for empty products.
        strided = torch.empty(2 * scale_b.numel(), dtype=scale_b.dtype, device="cuda")
        strided = strided[::2].copy_(scale_b)
        bad_scales = {
            "elements": (scale_b[:-1], scale_b),
            "contiguous scale": (strided, None),
            "on cuda": (scale_b.cpu(), scale_b[:0].cpu()),
            "float8_e8m0fnu": (scale_b.float(), scale_b[:0].float()),
        }
        for message, (bad, bad_for_zero_k) in bad_scales.items():
            with self.subTest(message):
                out = torch.empty(0, dtype=torch.bfloat16, device="cuda")
                with self.assertRaisesRegex(ValueError, message):
                    op.out(a, b, scale_a, bad, torch.bfloat16, out=out)
                self.assertEqual(out.shape, (0,))
                if bad_for_zero_k is not None:
                    with self.assertRaisesRegex(ValueError, message):
                        op(a[:, :0], b[:0], scale_a[:0], bad_for_zero_k, torch.bfloat16)
        # Freshly allocated K=0 operands have degenerate strides.
        empty_a = torch.empty(128, 0, dtype=a.dtype, device="cuda")
        empty_b = torch.empty(0, 256, dtype=a.dtype, device="cuda").t().contiguous().t()
        none = torch.empty(0, dtype=scale_a.dtype, device="cuda")
        self.assertEqual(
            op(empty_a, empty_b, none, none, torch.bfloat16),
            torch.zeros(128, 256, device="cuda", dtype=torch.bfloat16),
        )

    def test_single_row_operands(self):
        # A single row's stride is arbitrary, e.g. (1, 1) from torch.empty(K/4, 1).
        a, b, scale_a, scale_b, reference = _operands(128, 1, 512)
        b = torch.empty(b.shape, dtype=b.dtype, device="cuda").copy_(b)
        self.assertEqual(b.stride(), (1, 1))
        self.assertMatches(_scaled_mm(a, b, scale_a, scale_b), reference)

    def test_compile_dynamic(self):
        from torch._dynamo.testing import CompileCounterWithBackend

        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(_scaled_mm, backend=counter, dynamic=True)
        # 200 and 384 also change the padded scale size. K/4 = 160 stays apart
        # from M and N so duck sizing does not tie them.
        for m in (128, 200, 384):
            a, b, scale_a, scale_b, reference = _operands(m, 256, 640)
            self.assertMatches(compiled(a, b, scale_a, scale_b), reference)
        self.assertEqual(counter.frame_count, 1)

    def test_fake_rejects_bias(self):
        a, b, scale_a, scale_b, _ = _operands(128, 256, 512)
        bias = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
        with FakeTensorMode() as mode:
            args = [mode.from_tensor(t) for t in (a, b, scale_a, scale_b)]
            with self.assertRaisesRegex(NotImplementedError, "bias"):
                _scaled_mm(*args, bias=mode.from_tensor(bias))

    def test_contraction_dim(self):
        a, b, scale_a, scale_b, reference = _operands(128, 256, 512)
        for contraction_dim in ((1, 0), (-1, -2)):
            out = _scaled_mm(a, b, scale_a, scale_b, contraction_dim=contraction_dim)
            self.assertMatches(out, reference)

    def test_compile(self):
        a, b, scale_a, scale_b, reference = _operands(128, 256, 512)

        def fn(a, b, scale_a, scale_b):
            out = _scaled_mm(a, b, scale_a, scale_b)
            # Branch on the dtype so a wrong traced dtype changes the result.
            return out * 2 if out.dtype is torch.bfloat16 else out

        compiled = torch.compile(fn, fullgraph=True)
        self.assertMatches(compiled(a, b, scale_a, scale_b), reference * 2)

    def test_fake_out(self):
        a, b, scale_a, scale_b, _ = _operands(128, 256, 512)
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            args = _scaled_mm_v2_args(
                *map(mode.from_tensor, (a, b, scale_a, scale_b)), torch.bfloat16
            )
            out = torch.empty(0, dtype=torch.bfloat16, device="cuda")
            torch.ops.aten._scaled_mm_v2.out(*args, out=out)
            self.assertEqual(out.shape, (128, 256))

    def test_invalid_calls_raise(self):
        a, b, scale_a, scale_b, _ = _operands(128, 256, 512)
        fp8 = torch.zeros(128, 256, device="cuda", dtype=torch.float8_e4m3fn)
        cases = {
            "contiguous along K": lambda: _scaled_mm(
                a, b.contiguous(), scale_a, scale_b
            ),
            "SWIZZLE_32_4_4": lambda: _scaled_mm(
                a, b, scale_a, scale_b, swizzle_b=F.SwizzleType.NO_SWIZZLE
            ),
            "both operands": lambda: _scaled_mm(a, fp8, scale_a, scale_b),
            "multiple of 128": lambda: _scaled_mm(a[:, :16], b[:16], scale_a, scale_b),
            "scale_b to have": lambda: _scaled_mm(a, b, scale_a, scale_b[:-1]),
            "out_dtype": lambda: _scaled_mm(a, b, scale_a, scale_b, output_dtype=None),
        }
        for message, call in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                call()
        with self.assertRaisesRegex(NotImplementedError, "bias"):
            _scaled_mm(
                a,
                b,
                scale_a,
                scale_b,
                bias=torch.zeros(256, device="cuda", dtype=torch.bfloat16),
            )

    def test_disabled_override_reports_it(self):
        a, b, scale_a, scale_b, _ = _operands(128, 256, 512)
        with (
            torch.backends.python_native.cutedsl.disabled(),
            torch._native._unconditional_masked(),
            self.assertRaisesRegex(NotImplementedError, "CuTeDSL override"),
        ):
            _scaled_mm(a, b, scale_a, scale_b)


class TestScaledMmMXFP6CPU(TestCase):
    hw_classification = HardwareClassification.CPU

    def test_cpu_raises(self):
        a = torch.zeros(128, 96, dtype=torch.uint8).view(torch.float6_e2m3fn_x4)
        b = torch.zeros(256, 96, dtype=torch.uint8).view(torch.float6_e2m3fn_x4).t()
        scale_a = torch.zeros(128 * 4, dtype=torch.float8_e8m0fnu)
        scale_b = torch.zeros(256 * 4, dtype=torch.float8_e8m0fnu)
        with self.assertRaisesRegex(NotImplementedError, "NVIDIA CUDA device"):
            _scaled_mm(a, b, scale_a, scale_b, output_dtype=torch.bfloat16)


instantiate_parametrized_tests(TestScaledMmMXFP6CuTeDSL)


if __name__ == "__main__":
    run_tests()
