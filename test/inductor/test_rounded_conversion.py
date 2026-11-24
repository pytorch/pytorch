# Owner(s): ["module: inductor"]

import glob
import math
import os
import re
import unittest

import torch
from torch._higher_order_ops.associative_scan import associative_scan
from torch._inductor.cpu_vec_isa import valid_vec_isa_list
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import (
    fresh_cache,
    is_triton_fp8_dtype_supported,
    run_and_get_code,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCPUIf,
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
FALLBACK = "= torch.ops.aten._to_dtype_rounded.default("


def _pair_name(pair):
    src, dst = pair
    return f"{str(src).removeprefix('torch.')}_to_{str(dst).removeprefix('torch.')}"


def _lowered(device, dst):
    """Whether Inductor generates code for the conversion rather than falling back."""
    device = torch.device(device)
    # Inductor does not support float8_e5m3fnu yet.
    if dst == torch.float8_e5m3fnu:
        return False
    if device.type == "cpu":
        return True
    # Triton, on CUDA and ROCm, except for the fnuz dtypes (and on ROCm, any
    # float8 dtype but float8_e8m0fnu).
    unsupported = (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz)
    if torch.version.hip is not None:
        unsupported += (torch.float8_e4m3fn, torch.float8_e5m2)
    return (
        device.type == "cuda"
        and is_triton_fp8_dtype_supported(dst, device)
        and dst not in unsupported
    )


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
                if _lowered(device, dst):
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

    @parametrize(
        "dst",
        [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2],
        name_fn=lambda dst: str(dst).removeprefix("torch."),
    )
    def test_float64_to_dtype_rounds_once(self, device, dst):
        # Like eager, rounding to float32 first (the compute type of Triton, or
        # through float constructors in C++) must be avoided: test the
        # neighbors of the midpoints between dst's values.
        if not is_triton_fp8_dtype_supported(dst, device):
            self.skipTest(f"Triton does not support {dst} here")
        int_dtype = torch.int16 if dst.itemsize == 2 else torch.uint8
        bits = torch.arange(1 << (8 * dst.itemsize), dtype=torch.int32)
        values = bits.to(int_dtype).view(dst).double()
        values = values[values.isfinite()].unique()
        mid = (values[1:] + values[:-1]) / 2
        near = torch.cat([mid.nextafter(values[1:]), mid.nextafter(values[:-1])])
        x = torch.cat([near, -near, _inputs(torch.float64)]).to(device)
        if dst == torch.float8_e5m2 and torch.device(device).type == "cuda":
            # Triton's conversion to float8_e5m2 saturates, unlike eager's.
            x = x.clamp(-57344, 57344)

        def fn(x):
            return x.to(dst)

        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertSameValues(actual, fn(x))
        # The vectorized loop uses VecConvert, the tail c10::convert.
        if torch.device(device).type == "cpu" and valid_vec_isa_list():
            self.assertIn("at::vec::convert", "\n".join(code))

    def test_memory_format(self, device):
        def fn(x):
            return (x * 2).to(
                torch.bfloat16,
                round="toward_zero",
                memory_format=torch.channels_last,
            )

        x = torch.randn(2, 8, 5, 6, device=device)
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertNotIn(FALLBACK, "\n".join(code))
        self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last))
        self.assertSameValues(actual, fn(x))

    @parametrize(
        "dst",
        [torch.float16, torch.bfloat16],
        name_fn=lambda dst: str(dst).removeprefix("torch."),
    )
    def test_low_precision_users(self, device, dst):
        # The rounded result feeds ops that mix it with other values (in C++,
        # the low-precision legalization widens it).
        def fn(x, m):
            r = x.to(dst, round="toward_zero")
            return torch.maximum(r + r, r), torch.where(m, r * r, r)

        x = torch.randn(1024, device=device)
        m = torch.rand(1024, device=device) > 0.5
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), x, m)
        code = "\n".join(code)
        self.assertNotIn(FALLBACK, code)
        if torch.device(device).type == "cpu" and valid_vec_isa_list():
            # Only the conversion is scalarized in the vectorized loop.
            self.assertIn("at::vec::Vectorized", code)
        for a, e in zip(actual, fn(x, m)):
            self.assertSameValues(a, e)

    def test_fusion(self, device):
        # Before sm_89, Triton has no float8_e4m3fn.
        if is_triton_fp8_dtype_supported(torch.float8_e4m3fn, device):
            q_dtype = torch.float8_e4m3fn
        else:
            q_dtype = torch.float8_e5m2

        def fn(x, y):
            scale = x.abs().amax(dim=-1, keepdim=True) / 448
            e8m0 = scale.to(torch.float8_e8m0fnu, round="toward_positive")
            q = (x / e8m0.float()).to(q_dtype, round="toward_zero")
            return q, e8m0, (y * 2).to(torch.bfloat16, round="toward_negative") + 1

        x = torch.randn(64, 32, device=device)
        y = torch.randn(1024, device=device)
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x, y)
        for a, e in zip(actual, fn(x, y)):
            self.assertSameValues(a, e)
        self.assertNotIn(FALLBACK, code)
        # The conversions fuse with their producers (kernel names list the
        # fused ops, sorted): the quantization (_to_copy, div) with the amax.
        self.assertRegex(code, r"fused__to_copy__to_dtype_rounded_abs_\w*amax_div")
        self.assertRegex(code, r"fused_\w*_to_dtype_rounded_\w*add_\w*mul")

    @skipCPUIf(True, "tests a Triton helper")
    def test_round_toward_to_odd(self, device):
        import triton
        import triton.language as tl

        from torch._inductor.runtime import triton_helpers

        @triton.jit
        def kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(in_ptr + offsets, mask=mask)
            y = triton_helpers.round_toward(x, tl.float32, "to_odd", 0.0)
            tl.store(out_ptr + offsets, y, mask=mask)

        x = torch.cat(
            [
                _inputs(torch.float64),
                torch.tensor(
                    [3.5e38, -3.5e38, 3.4028235677973366e38, 1e-46],
                    dtype=torch.float64,
                ),
                torch.randn(1 << 16, dtype=torch.float64),
            ]
        ).to(device)
        out = torch.empty_like(x, dtype=torch.float32)
        kernel[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
        # Truncate, and set the last bit if inexact.
        rz = x.cpu().to(torch.float32, round="toward_zero")
        inexact = (rz.double() != x.cpu()).to(torch.int32)
        expected = (rz.view(torch.int32) | inexact).view(torch.float32)
        self.assertSameValues(out, expected)

    @skipCPUIf(True, "tests Triton helpers")
    def test_rounding_helpers(self, device):
        # The Triton helpers of the conversions without a dedicated
        # instruction, in every mode, even where Inductor would emit one.
        import triton
        import triton.language as tl

        from torch._inductor.runtime import triton_helpers

        @triton.jit
        def kernel(
            in_ptr, out_ptr, n, DTYPE: tl.constexpr, MODE: tl.constexpr,
            SATURATION: tl.constexpr, BLOCK: tl.constexpr,
        ):  # fmt: skip
            offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
            mask = offsets < n
            x = tl.load(in_ptr + offsets, mask=mask)
            if DTYPE == tl.uint8:
                y = triton_helpers.float32_to_float8_e8m0(x, MODE)
            else:
                y = triton_helpers.round_toward(x, DTYPE, MODE, SATURATION)
                if DTYPE == tl.float8e5:
                    y = triton_helpers.float8_e5m2_infinities(x, y, MODE)
            tl.store(out_ptr + offsets, y, mask=mask)

        pairs = [
            (torch.float64, torch.float32, tl.float32),
            (torch.float64, torch.float16, tl.float16),
            (torch.float64, torch.bfloat16, tl.bfloat16),
            (torch.float32, torch.float16, tl.float16),
            (torch.float32, torch.bfloat16, tl.bfloat16),
            (torch.float32, torch.float8_e8m0fnu, tl.uint8),
        ]
        if _lowered(device, torch.float8_e5m2):
            pairs.append((torch.float32, torch.float8_e5m2, tl.float8e5))
        if _lowered(device, torch.float8_e4m3fn):
            pairs.append((torch.float32, torch.float8_e4m3fn, tl.float8e4nv))
        for src, dst, tl_dtype in pairs:
            x = _inputs(src, 1 << 20)
            if src == torch.float32:
                # Every float32 value below 2^-125, subnormals included.
                x = torch.cat(
                    [x, torch.arange(0, 1 << 24, dtype=torch.int32).view(src)]
                )
            x = x.to(device)
            for mode in MODES:
                with self.subTest(src=src, dst=dst, mode=mode):
                    out_dtype = torch.uint8 if dst == torch.float8_e8m0fnu else dst
                    out = torch.empty_like(x, dtype=out_dtype)
                    kernel[(triton.cdiv(x.numel(), 1024),)](
                        x, out, x.numel(), tl_dtype, mode,
                        448.0 if dst == torch.float8_e4m3fn else 0.0, BLOCK=1024,
                    )  # fmt: skip
                    actual = out.view(dst)
                    self.assertSameValues(actual, x.cpu().to(dst, round=mode))


def _compile_and_get_ptx(fn, *args):
    """Compiles and runs fn with Inductor. Returns its result and the PTX of its
    Triton kernels (one per config), which Triton writes to its cache."""
    with fresh_cache():
        torch._dynamo.reset()
        result = torch.compile(fn, fullgraph=True)(*args)
        cache_dir = os.environ["TRITON_CACHE_DIR"]
        paths = glob.glob(os.path.join(cache_dir, "**", "*.ptx"), recursive=True)
        ptx = []
        for path in paths:
            with open(path) as f:
                ptx.append(f.read())
        return result, ptx


# A packed instruction, of two registers (rather than of a constant).
PACKED = r"\s+%\w+, %\w+, %\w+;"
# The directed instructions, to check that a conversion steps the result of the
# conversion to nearest.
DIRECTED = r"cvt\.r[zpm]\."


@unittest.skipIf(not HAS_CUDA, "requires CUDA")
class TestRoundedConversionInductorCUDAOnly(_RoundedConversionTestMixin, TestCase):
    hw_classification = HardwareClassification.CUDA

    def assertPTX(self, ptx, present, absent=None):
        if present is not None:
            self.assertTrue(
                any(re.search(present, p) for p in ptx), f"{present} not in PTX"
            )
        if absent is not None:
            self.assertFalse(any(re.search(absent, p) for p in ptx), f"{absent} in PTX")

    def test_cuda_instructions(self):
        # The instructions of torch._inductor.codegen.triton.CVT, packed where
        # they have a packed form, and otherwise Triton's conversion to nearest
        # without a directed instruction (triton_helpers.round_toward).
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
        # (src, dst, mode, regex in some PTX, regex in none)
        expectations = [
            (f32, f16, "toward_positive", r"cvt\.rp\.f16\.f32", None),
            (f32, f16, "toward_negative", r"cvt\.rm\.f16\.f32", None),
            (f64, f32, "toward_positive", r"cvt\.rp\.f32\.f64", None),
            (f64, f16, "to_nearest_even", r"cvt\.rn\.f16\.f64", None),
            (f64, f16, "toward_zero", r"cvt\.rz\.f16\.f64", None),
            (f32, e8m0, "to_nearest_even", None, "ue8m0"),
        ]
        if cc >= 80:
            expectations += [
                (f32, f16, "to_nearest_even", r"cvt\.rn\.f16x2\.f32" + PACKED, None),
                (f32, f16, "toward_zero", r"cvt\.rz\.f16x2\.f32" + PACKED, None),
                (f32, bf16, "to_nearest_even", r"cvt\.rn\.bf16x2\.f32" + PACKED, None),
                (f32, bf16, "toward_zero", r"cvt\.rz\.bf16x2\.f32" + PACKED, None),
                (f32, bf16, "toward_negative", r"cvt\.rm\.bf16\.f32", None),
            ]
        else:
            expectations.append((f32, f16, "toward_zero", r"cvt\.rz\.f16\.f32", None))
        if cc >= 89:
            rn_e4m3 = r"cvt\.rn\.satfinite\.e4m3x2\.f32" + PACKED
            rn_e5m2 = r"cvt\.rn\.satfinite\.e5m2x2\.f32" + PACKED
            expectations += [
                (f32, e4m3, "to_nearest_even", rn_e4m3, None),
                (f32, e4m3, "toward_positive", rn_e4m3, DIRECTED),
                (f32, e5m2, "to_nearest_even", rn_e5m2, None),
                (f32, e5m2, "toward_negative", rn_e5m2, DIRECTED),
                # Through float32, rounded to odd or in the same direction.
                (f64, e4m3, "to_nearest_even", r"cvt\.rn\.f32\.f64", None),
                (f64, e5m2, "toward_zero", r"cvt\.rz\.f32\.f64", None),
            ]
            if cc == 107:
                rz_e4m3 = r"cvt\.rz\.satfinite\.e4m3x2\.f32" + PACKED
                rz_e5m2 = r"cvt\.rz\.satfinite\.e5m2x2\.f32" + PACKED
                expectations += [
                    (f32, e4m3, "toward_zero", rz_e4m3, None),
                    (f32, e5m2, "toward_zero", rz_e5m2, None),
                ]
            else:
                expectations += [
                    (f32, e4m3, "toward_zero", rn_e4m3, DIRECTED),
                    (f32, e5m2, "toward_zero", rn_e5m2, DIRECTED),
                ]
        if cc >= 90:
            expectations.append(
                (f64, bf16, "toward_positive", r"cvt\.rp\.bf16\.f64", None)
            )
        if cc >= 100:
            rp_e8m0 = r"cvt\.rp\.ue8m0x2\.f32" + PACKED
            rz_e8m0 = r"cvt\.rz\.ue8m0x2\.f32" + PACKED
            expectations += [
                (f32, e8m0, "toward_positive", rp_e8m0, None),
                (f32, e8m0, "toward_zero", rz_e8m0, None),
                # Unsigned, so the same as toward_zero.
                (f32, e8m0, "toward_negative", rz_e8m0, None),
                (f64, e8m0, "toward_positive", r"cvt\.rp\.f32\.f64", None),
            ]
        else:
            expectations.append((f32, e8m0, "toward_positive", None, "ue8m0"))
        for src, dst, mode, present, absent in expectations:
            with self.subTest(src=src, dst=dst, mode=mode):
                x = _inputs(src).cuda()

                def fn(x):
                    return x.to(dst, round=mode)

                actual, ptx = _compile_and_get_ptx(fn, x)
                self.assertSameValues(actual, fn(x))
                self.assertTrue(ptx, "no PTX")
                self.assertPTX(ptx, present, absent)

    def test_fused_instructions(self):
        # Packed in a kernel fused with a reduction.
        if _cuda_capability() < 100:
            self.skipTest("cvt.rp.ue8m0x2.f32 needs sm_100")

        def fn(x):
            scale = x.abs().amax(dim=-1, keepdim=True)
            return scale, (x - scale).to(torch.float8_e8m0fnu, round="toward_positive")

        x = torch.randn(1024, 512, device="cuda")
        (scale, actual), ptx = _compile_and_get_ptx(fn, x)
        self.assertSameValues(
            actual, (x - scale).to(torch.float8_e8m0fnu, round="toward_positive")
        )
        fused = [p for p in ptx if "max.f32" in p or "max.NaN.f32" in p]
        self.assertPTX(fused, r"cvt\.rp\.ue8m0x2\.f32" + PACKED)

    def test_float64_instructions(self):
        # The default conversions from float64, as in eager.
        cc = _cuda_capability()
        rn_f32 = r"cvt\.rn\.f32\.f64"
        expectations = [
            (torch.float16, r"cvt\.rn\.f16\.f64"),
            (torch.float8_e5m2, rn_f32),
        ]
        if cc >= 89:
            expectations.append((torch.float8_e4m3fn, rn_f32))
        if cc >= 90:
            expectations.append((torch.bfloat16, r"cvt\.rn\.bf16\.f64"))

        x = torch.randn(1024, device="cuda", dtype=torch.float64)
        for dst, present in expectations:
            with self.subTest(dst=dst):

                def fn(x):
                    return x.to(dst)

                actual, ptx = _compile_and_get_ptx(fn, x)
                self.assertSameValues(actual, fn(x))
                self.assertPTX(ptx, present)

    def test_scan_combine_fn(self):
        # Scans lift their combine function into a separate Triton function,
        # so the float64 conversion, routed through to_dtype_rounded, must not
        # write to the kernel. (round= cannot appear there: combine functions
        # may only use ops tagged pointwise.) This combine function is
        # associative, as the rounding is monotonic and idempotent.
        convert = torch.ops.prims.convert_element_type

        def combine(a, b):
            # Back through float32: Triton has no float8 -> float64 conversion.
            e5m2 = convert(b, torch.float8_e5m2)
            return torch.maximum(a, convert(convert(e5m2, torch.float32), a.dtype))

        def fn(x):
            return associative_scan(combine, x, 1, combine_mode="pointwise")

        x = torch.randn(4, 64, device="cuda", dtype=torch.float64) * 100
        actual, (code,) = run_and_get_code(torch.compile(fn, fullgraph=True), x)
        self.assertIn("_triton_helper_fn", code)
        self.assertIn('"to_odd"', code)
        self.assertEqual(actual, fn(x), atol=0, rtol=0)


# Eager has no _to_dtype_rounded kernel for MPS or XPU, to compare with.
instantiate_device_type_tests(
    TestRoundedConversionInductor, globals(), except_for=("mps", "xpu")
)

if __name__ == "__main__":
    run_tests()
