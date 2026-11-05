# Owner(s): ["module: inductor"]

import math
import unittest

import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._inductor.cpu_vec_isa import valid_vec_isa_list
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import is_triton_fp8_dtype_supported, run_and_get_code
from torch.testing._internal.common_device_type import (
    expectedFailureIfRocm,
    expectedFailureXPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import HardwareClassification, parametrize
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


MODES = ("to_nearest_even", "toward_zero", "toward_positive", "toward_negative")
FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
)
PAIRS = [
    (torch.float64, torch.float32),
    (torch.float64, torch.float16),
    (torch.float64, torch.bfloat16),
    (torch.float64, torch.float8_e4m3fn),
    (torch.float64, torch.float8_e8m0fnu),
    (torch.float32, torch.float16),
    (torch.float32, torch.bfloat16),
    *[(torch.float32, dtype) for dtype in FLOAT8_DTYPES],
    (torch.float16, torch.bfloat16),
    (torch.float16, torch.float8_e5m2),
    (torch.bfloat16, torch.float16),
    (torch.bfloat16, torch.float8_e8m0fnu),
]

HAS_CUDA = HAS_GPU and GPU_TYPE == "cuda" and torch.version.hip is None
FALLBACK = "= torch.ops.aten.round_to.default("


def _pair_name(pair):
    src, dst = pair
    return f"{str(src).removeprefix('torch.')}_to_{str(dst).removeprefix('torch.')}"


def _lowered(device, src, dst):
    """Whether Inductor generates code for round_to rather than falling back."""
    device = torch.device(device)
    # Inductor does not support float8_e5m3fnu yet.
    if dst == torch.float8_e5m3fnu:
        return False
    if device.type == "cpu":
        return True
    # Triton reduces float64 to float32 with PTX.
    if src == torch.float64 and (device.type != "cuda" or torch.version.hip):
        return False
    return is_triton_fp8_dtype_supported(dst, device)


def _cuda_capability():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _inputs(src, n=1 << 16):
    """Random bit patterns biased toward the narrow formats' range, the values
    of a fine grid around powers of two, and special values."""
    if src in (torch.float16, torch.bfloat16):
        return (
            torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(src)
        )
    if src is torch.float32:
        bits = torch.randint(-(2**31), 2**31 - 1, (n,), dtype=torch.int32)
        x = bits.view(torch.float32)
    else:
        exponents = torch.randint(1023 - 160, 1023 + 160, (n,), dtype=torch.int64)
        mantissas = torch.randint(0, 2**52, (n,), dtype=torch.int64)
        signs = torch.randint(0, 2, (n,), dtype=torch.int64) << 63
        x = (signs | (exponents << 52) | mantissas).view(torch.float64)
    grid = torch.cat(
        [
            torch.linspace(0.5, 4, 4097, dtype=torch.float64) * 2.0**e
            for e in (-140, -127, -24, -9, -6, 0, 7, 8, 15, 16, 127)
        ]
    ).to(src)
    finfo = torch.finfo(src)
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            math.inf,
            -math.inf,
            math.nan,
            finfo.max,
            -finfo.max,
            finfo.tiny,
            finfo.tiny / 3,
        ],
        dtype=src,
    )
    return torch.cat([x, grid, -grid, specials])


class _RoundedConversionTestMixin:
    def assertSameValues(self, actual, expected):
        actual = actual.cpu().double()
        expected = expected.cpu().double()
        self.assertEqual(actual.isnan(), expected.isnan())
        ok = expected.isnan() | (
            (actual == expected) & (torch.signbit(actual) == torch.signbit(expected))
        )
        self.assertTrue(ok.all(), f"{int((~ok).sum())} mismatches")

    def _check(self, device, src, dst, mode):
        x = _inputs(src).to(device)

        def fn(x):
            return x.to(dst, round=mode)

        torch._dynamo.reset()
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertEqual(actual.dtype, dst)
        self.assertSameValues(actual, fn(x))
        return "\n".join(code)


class TestRoundedConversionInductor(_RoundedConversionTestMixin, TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("pair", PAIRS, name_fn=_pair_name)
    def test_matches_eager(self, device, pair):
        src, dst = pair
        for mode in MODES:
            with self.subTest(mode=mode):
                code = self._check(device, src, dst, mode)
                if _lowered(device, src, dst):
                    self.assertNotIn(FALLBACK, code)
                else:
                    self.assertIn(FALLBACK, code)
                if torch.device(device).type == "cpu":
                    self.assertIn("c10::convert_with_rounding", code)

    def test_e5m3fnu_falls_back(self, device):
        for mode in MODES:
            with self.subTest(mode=mode):
                code = self._check(device, torch.float32, torch.float8_e5m3fnu, mode)
                self.assertIn(FALLBACK, code)

    # Triton reduces float64 with PTX, so elsewhere it still rounds twice.
    @expectedFailureIfRocm
    @expectedFailureXPU
    def test_float64_to_dtype_rounds_once(self, device):
        # Like eager, rounding to float32 first (the compute type of Triton, or
        # through float constructors in C++) must be avoided: test the
        # neighbors of the midpoints between dst's values.
        for dst in (
            torch.float16,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            if not is_triton_fp8_dtype_supported(dst, device):
                # The conversion falls back to eager.
                continue
            int_dtype = torch.int16 if dst.itemsize == 2 else torch.uint8
            bits = torch.arange(1 << (8 * dst.itemsize), dtype=torch.int32)
            values = bits.to(int_dtype).view(dst).double()
            values = values[values.isfinite()].unique()
            mid = (values[1:] + values[:-1]) / 2
            near = torch.cat([mid.nextafter(values[1:]), mid.nextafter(values[:-1])])
            x = torch.cat([near, -near, _inputs(torch.float64)]).to(device)

            def fn(x):
                return x.to(dst)

            with self.subTest(dst=dst):
                torch._dynamo.reset()
                actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
                self.assertSameValues(actual, fn(x))
                code = "\n".join(code)
                if torch.device(device).type == "cpu":
                    # The vectorized loop uses VecConvert, the tail c10::convert.
                    if valid_vec_isa_list():
                        self.assertIn("at::vec::convert", code)
                else:
                    self.assertTrue(
                        "cvt.rn.f16.f64" in code
                        or "cvt.rn.bf16.f64" in code
                        or "round_float64_to_odd_float32" in code
                    )


@unittest.skipIf(not HAS_CUDA, "requires CUDA")
class TestRoundedConversionInductorCUDA(_RoundedConversionTestMixin, TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_cuda_instructions(self):
        cc = _cuda_capability()
        f32, f64, f16, bf16 = (
            torch.float32,
            torch.float64,
            torch.float16,
            torch.bfloat16,
        )
        e4m3, e5m2, e8m0 = (
            torch.float8_e4m3fn,
            torch.float8_e5m2,
            torch.float8_e8m0fnu,
        )
        expectations = [
            (f32, f16, "to_nearest_even", "cvt.rn.f16.f32"),
            (f32, f16, "toward_zero", "cvt.rz.f16.f32"),
            (f32, f16, "toward_positive", "cvt.rp.f16.f32"),
            (f32, f16, "toward_negative", "cvt.rm.f16.f32"),
            (f64, f32, "toward_positive", "cvt.rp.f32.f64"),
            (f64, f16, "toward_zero", "cvt.rz.f16.f64"),
            # Round to odd through float32.
            (f64, e5m2, "toward_zero", "round_float64_to_odd_float32"),
            (f32, e8m0, "to_nearest_even", "round_float32_to_bits"),
        ]
        if cc >= 80:
            expectations += [
                (f32, bf16, "to_nearest_even", "cvt.rn.bf16.f32"),
                (f32, bf16, "toward_zero", "cvt.rz.bf16.f32"),
            ]
        if cc >= 89:
            # Round to nearest, then step toward x in the directed modes.
            expectations += [
                (f32, e4m3, "to_nearest_even", "float32_to_float8_bits"),
                (f32, e5m2, "toward_positive", "float32_to_float8_bits"),
            ]
        else:
            expectations.append((f32, e5m2, "toward_zero", "round_float32_to_bits"))
        if cc >= 90:
            expectations += [
                (f32, bf16, "toward_negative", "cvt.rm.bf16.f32"),
                (f64, bf16, "toward_positive", "cvt.rp.bf16.f64"),
            ]
        else:
            expectations.append(
                (f64, bf16, "toward_zero", "round_float64_to_odd_float32")
            )
        if cc >= 100:
            expectations += [
                (f32, e8m0, "toward_positive", "cvt.rp.ue8m0x2.f32"),
                (f32, e8m0, "toward_zero", "cvt.rz.ue8m0x2.f32"),
                (f32, e8m0, "toward_negative", "cvt.rz.ue8m0x2.f32"),
            ]
        for src, dst, mode, snippet in expectations:
            if not is_triton_fp8_dtype_supported(dst, "cuda"):
                continue
            with self.subTest(src=src, dst=dst, mode=mode):
                code = self._check("cuda", src, dst, mode)
                self.assertIn(snippet, code)
                self.assertNotIn("fp_downcast_rounding", code)
                if dst == e8m0:
                    self.assertNotIn("satfinite", code)

    def test_memory_format(self):
        def fn(x):
            return (x * 2).to(
                torch.bfloat16,
                round="toward_zero",
                memory_format=torch.channels_last,
            )

        x = torch.randn(2, 8, 5, 6, device="cuda")
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertNotIn(FALLBACK, "\n".join(code))
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))
        self.assertSameValues(actual, fn(x))

    def test_fusion(self):
        # Before sm_89, Triton has no float8_e4m3fn.
        if is_triton_fp8_dtype_supported(torch.float8_e4m3fn, "cuda"):
            q_dtype = torch.float8_e4m3fn
        else:
            q_dtype = torch.float8_e5m2

        def fn(x, y):
            scale = x.abs().amax(dim=-1, keepdim=True) / 448
            e8m0 = scale.to(torch.float8_e8m0fnu, round="toward_positive")
            q = (x / e8m0.float()).to(q_dtype, round="toward_zero")
            return q, e8m0, (y * 2).to(torch.bfloat16, round="toward_negative") + 1

        x = torch.randn(64, 32, device="cuda")
        y = torch.randn(1024, device="cuda")
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x, y)
        for a, e in zip(actual, fn(x, y)):
            self.assertSameValues(a, e)
        self.assertNotIn("extern_kernels", code)
        self.assertNotIn(FALLBACK, code)

    def test_scan_combine_fn(self):
        # Scans lift their combine function into a separate Triton function,
        # so the conversions must not write to the kernel. This combine
        # function is associative, as the rounding is monotonic and idempotent.
        for dtype, round in (
            (torch.float32, "toward_zero"),
            (torch.float64, None),
            (torch.float64, "toward_positive"),
        ):

            def combine(a, b):
                return torch.maximum(a, b.to(torch.float8_e5m2, round=round).to(dtype))

            def fn(x):
                return associative_scan(combine, x, 1, combine_mode="pointwise")

            x = torch.randn(4, 64, device="cuda", dtype=dtype) * 100
            with self.subTest(dtype=dtype, round=round):
                torch._dynamo.reset()
                actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
                self.assertIn("_triton_helper_fn", code)
                self.assertEqual(actual, fn(x), atol=0, rtol=0)

    def test_software_helper(self):
        # Exercise the Triton software path for every format and mode, even
        # where Inductor would emit a hardware instruction.
        import triton
        import triton.language as tl

        from torch._inductor.codegen.triton import ROUNDING_FORMATS, ROUNDING_MODES
        from torch._inductor.runtime import triton_helpers

        @triton.jit
        def kernel(
            in_ptr, out_ptr, n, MODE: tl.constexpr, MANT_BITS: tl.constexpr,
            BIAS: tl.constexpr, SIGNED: tl.constexpr, HAS_NEG_ZERO: tl.constexpr,
            HAS_SUBNORMALS: tl.constexpr, SIGN_BIT: tl.constexpr,
            MAX_FINITE: tl.constexpr, OVERFLOW: tl.constexpr, NAN: tl.constexpr,
            BLOCK: tl.constexpr,
        ):  # fmt: skip
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(in_ptr + offsets, mask=mask)
            bits = triton_helpers.round_float32_to_bits(
                x, MODE, MANT_BITS, BIAS, SIGNED, HAS_NEG_ZERO, HAS_SUBNORMALS,
                SIGN_BIT, MAX_FINITE, OVERFLOW, NAN,
            )  # fmt: skip
            tl.store(out_ptr + offsets, bits, mask=mask)

        x = torch.cat(
            [
                _inputs(torch.float32, 1 << 20),
                torch.arange(0, 1 << 24, dtype=torch.int32).view(torch.float32),
            ]
        ).cuda()
        for dst, fmt in ROUNDING_FORMATS.items():
            for mode in MODES:
                with self.subTest(dst=dst, mode=mode):
                    out = torch.empty_like(x, dtype=torch.int32)
                    kernel[(triton.cdiv(x.numel(), 1024),)](
                        x, out, x.numel(), ROUNDING_MODES[mode], *fmt, BLOCK=1024
                    )
                    bits = torch.finfo(dst).bits
                    int_dtype = torch.int16 if bits == 16 else torch.uint8
                    actual = out.to(int_dtype).view(dst)
                    self.assertSameValues(actual, x.cpu().to(dst, round=mode))


instantiate_device_type_tests(
    TestRoundedConversionInductor, globals(), except_for=("mps",)
)

if __name__ == "__main__":
    run_tests()
