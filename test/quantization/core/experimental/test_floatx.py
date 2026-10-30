# Owner(s): ["oncall: quantization"]

import copy
import math
import struct
import unittest

import torch
from torch._prims_common import _uncastable_shell_dtypes
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    dtypesIfMPS,
    dtypesIfXPU,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    skipMPS,
)
from torch.testing._internal.common_quantized import (
    _float32_to_float6_e2m3fn_x4,
    _float6_e2m3fn_x4_to_float32,
    pack_uint6,
    unpack_uint6,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    HardwareClassification,
    IS_WINDOWS,
    parametrize,
    run_tests,
    subtest,
    TemporaryFileName,
    TestCase,
)


FLOAT8_DTYPES = [
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e8m0fnu,
    torch.float8_e5m3fnu,
]

CUDA_FLOAT8_DTYPES = [
    torch.float8_e5m2,
    torch.float8_e4m3fn,
    torch.float8_e8m0fnu,
    torch.float8_e5m3fnu,
]
MPS_FLOAT8_DTYPES = [torch.float8_e4m3fn]
# torch-xpu-ops has no copy kernel for float8_e5m3fnu yet
XPU_FLOAT8_DTYPES = [d for d in FLOAT8_DTYPES if d is not torch.float8_e5m3fnu]

# The following information are not yet provided by torch.finfo.

MANTISSA_BITS = {
    torch.float8_e5m2: 2,
    torch.float8_e5m2fnuz: 2,
    torch.float8_e4m3fn: 3,
    torch.float8_e4m3fnuz: 3,
    torch.float8_e8m0fnu: 0,
    torch.float8_e5m3fnu: 3,
}

# As in np.finfo(dtype).minexp
MINEXP = {
    torch.float8_e5m2: -14,
    torch.float8_e5m2fnuz: -15,
    torch.float8_e4m3fn: -6,
    torch.float8_e4m3fnuz: -7,
    torch.float8_e8m0fnu: -127,
    torch.float8_e5m3fnu: -14,
}

SPECIAL_NUMBERS = {
    torch.float8_e5m2: [
        ("01111100", float("inf"), "inf"),
        ("11111100", -1.0 * float("inf"), "neg_inf"),
        ("01111101", float("nan"), "nan"),
        ("11111101", float("nan"), "nan"),
        ("01111110", float("nan"), "nan"),
        ("11111110", float("nan"), "nan"),
        ("01111111", float("nan"), "nan"),
        ("11111111", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("10000000", -0.0, "neg_zero"),
        ("01111011", 57344.0, "max_normal"),
        ("11111011", -57344.0, "neg_max_normal"),
        ("00000100", 2**-14, "min_normal"),
        ("10000100", -1 * (2**-14), "neg_min_normal"),
        ("00000011", 0.75 * (2**-14), "max_subnorm"),
        ("10000011", -0.75 * (2**-14), "neg_max_subnorm"),
        ("00000001", 2**-16, "min_subnorm"),
        ("10000001", -1 * (2**-16), "neg_min_subnorm"),
    ],
    torch.float8_e5m2fnuz: [
        ("10000000", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("00000000", -0.0, "neg_zero"),
        ("01111111", 57344.0, "max_normal"),
        ("11111111", -57344.0, "neg_max_normal"),
        ("00000100", 2**-15, "min_normal"),
        ("10000100", -1 * (2**-15), "neg_min_normal"),
        ("00000011", 0.75 * (2**-15), "max_subnorm"),
        ("10000011", -0.75 * (2**-15), "neg_max_subnorm"),
        ("00000001", 0.25 * (2**-15), "min_subnorm"),
        ("10000001", -0.25 * (2**-15), "neg_min_subnorm"),
    ],
    torch.float8_e4m3fn: [
        ("01111111", float("nan"), "nan"),
        ("11111111", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("10000000", -0.0, "neg_zero"),
        ("01111110", 448.0, "max_normal"),
        ("11111110", -448.0, "neg_max_normal"),
        ("00001000", 2**-6, "min_normal"),
        ("10001000", -1 * (2**-6), "neg_min_normal"),
        ("00000111", 0.875 * (2**-6), "max_subnorm"),
        ("10000111", -0.875 * (2**-6), "neg_max_subnorm"),
        ("00000001", 2**-9, "min_subnorm"),
        ("10000001", -1 * (2**-9), "neg_min_subnorm"),
    ],
    torch.float8_e4m3fnuz: [
        ("10000000", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("00000000", -0.0, "neg_zero"),
        ("01111111", 240.0, "max_normal"),
        ("11111111", -240.0, "neg_max_normal"),
        ("00001000", 2**-7, "min_normal"),
        ("10001000", -1 * (2**-7), "neg_min_normal"),
        ("00000111", 0.875 * (2**-7), "max_subnorm"),
        ("10000111", -0.875 * (2**-7), "neg_max_subnorm"),
        ("00000001", 0.125 * (2**-7), "min_subnorm"),
        ("10000001", -0.125 * (2**-7), "neg_min_subnorm"),
    ],
    torch.float8_e8m0fnu: [
        ("00000000", float(2**-127), "smallest_number"),
        ("11111110", float(2**127), "largest_number"),
        ("01111110", 0.5, "zero_point_five"),
        ("01111111", 1.0, "one"),
        ("10000000", 2.0, "two"),
        ("11111111", float("nan"), "nan"),
    ],
    torch.float8_e5m3fnu: [
        ("00000000", 0.0, "zero"),
        ("00000001", 2**-17, "min_subnorm"),
        ("00000111", 0.875 * (2**-14), "max_subnorm"),
        ("00001000", 2**-14, "min_normal"),
        ("01111000", 1.0, "one"),
        ("11111000", 65536.0, "min_exponent_31"),
        ("11111110", 114688.0, "max_normal"),
        ("11111111", float("nan"), "nan"),
    ],
}

FLOAT8_DTYPES_WITH_INF = [torch.float8_e5m2]

FLOAT8_DTYPES_SATURATE_ON_OVERFLOW = [torch.float8_e4m3fn]


def _int_bits_to_float(x):
    y = struct.unpack("!f", struct.pack("!I", x))[0]
    return y


def simulate_fp8_precision(input, variant):
    """Round input (as float32) to the given float8 datatype variant."""

    # Constants
    dtype = torch.float32
    int_type = torch.int32
    mbits = MANTISSA_BITS[variant]
    minexp = MINEXP[variant]  # ml_dtypes.finfo(variant).

    input = input.to(dtype)

    # Extract bitfield components
    signs = torch.sign(input)
    input_int = torch.abs(input).view(int_type)

    exponent_bits = (input_int & 0x7F800000) >> 23
    mantissa_bits = input_int & 0x007FFFFF

    exponent_base = exponent_bits - 0x7F

    # Add implicit leading 1 to mantissas, i.e. create 1.mmmmmmmm
    f32_is_normal = exponent_bits != 0
    mantissa_val_base = f32_is_normal * 0x00800000 + mantissa_bits

    # Shift mantissa to match minimum exponent - denormals in the lower
    # precision dtype remain normal in the higher precision dtype
    denormal_bits = torch.maximum(
        minexp - exponent_base, torch.tensor(0, dtype=int_type)
    )
    mantissa_val = mantissa_val_base >> denormal_bits
    exponent = exponent_base + denormal_bits

    # Round off mantissas
    last_unrounded_bit = 1 << (23 - mbits)
    rounding_mask = last_unrounded_bit - 1
    mantissa_val_rounded = (mantissa_val + (rounding_mask >> 1)) & ~rounding_mask

    # Round ties to nearest even
    ties = (mantissa_val & rounding_mask) == (last_unrounded_bit >> 1)
    is_odd = (mantissa_val_rounded & last_unrounded_bit) != 0
    mantissa_val_rounded += (ties & is_odd) * last_unrounded_bit

    # Re-compose mantissa and exponent
    vals = (mantissa_val_rounded * 2.0 ** (-23 + exponent)).to(dtype)

    # Replace overflows: inf for types that have it, saturate to max for types
    # that use satfinite semantics, NaN otherwise
    overflow = vals > torch.finfo(variant).max
    if variant in FLOAT8_DTYPES_WITH_INF:
        vals[overflow] = torch.inf
    elif variant in FLOAT8_DTYPES_SATURATE_ON_OVERFLOW:
        vals[overflow] = torch.finfo(variant).max
    else:
        vals[overflow] = torch.nan

    return vals * signs


def _round_e8m0_rne(biased_exponent, lsb, g, r, s):
    round_up = False

    # apply g,r,s rounding rules for RNE rounding
    if g == 1:
        if (r == 1) or (s == 1):
            round_up = True
        else:
            if lsb:
                round_up = True

    # round up if necessary
    if round_up:
        biased_exponent += 1

    return biased_exponent


ROUND_TRIP_TEST_CASES = (
    # A general 'soak test'.
    subtest(
        lambda dtype, device: torch.rand((100, 100), device=device)
        * torch.finfo(dtype).max,
        name="soak",
    ),
    # A range below the smallest normal in the lower precision type, to ensure
    # these are rounded correctly to their nearest subnormal in that type.
    subtest(
        lambda dtype, device: torch.rand(1000, device=device)
        * 2
        * torch.finfo(dtype).smallest_normal,
        name="subnormals",
    ),
    # A range of integers to exert rounding to nearest even.
    subtest(
        lambda dtype, device: torch.arange(
            int(torch.finfo(dtype).max), dtype=torch.int, device=device
        ),
        name="rte",
    ),
    # Values around max.
    subtest(
        lambda dtype, device: torch.finfo(dtype).max
        + (torch.finfo(dtype).eps * torch.finfo(dtype).max)
        * torch.arange(-3, 3, 0.25, device=device),
        name="extremes",
    ),
)


class TestFloat8Dtype(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_creation_with_zeros(self, dtype, device):
        """Sanity test, round-trip casting of zeros."""
        x8 = torch.zeros(8, dtype=dtype, device=device)
        if dtype is torch.float8_e8m0fnu:
            # zeros are not supported for this dtype, values get clamped
            # to 2 ^ -127
            x = torch.full((8,), 2**-127, dtype=torch.float, device=device)
            self.assertEqual(x, x8.float(), atol=0, rtol=0)
        else:
            x = torch.zeros(8, dtype=torch.float, device=device)
            self.assertEqual(x, x8.float(), atol=0, rtol=0)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    @parametrize("get_input", ROUND_TRIP_TEST_CASES)
    def test_cast_round_trip(self, dtype, get_input, device):
        """Numerical test of float8 conversion, by performing a round-trip cast
        to the float8 dtype and back to float32, comparing against simulated
        lower precision."""
        if dtype in (torch.float8_e8m0fnu, torch.float8_e5m3fnu):
            self.skipTest(f"numerics for {dtype} are tested elsewhere")

        x = get_input(dtype, device)
        x = torch.cat((x, -x))
        x8 = x.to(dtype)
        x8_simulated = simulate_fp8_precision(x, dtype)
        self.assertEqual(x8_simulated, x8.float())

    @skipMPS
    def test_float8_e8m0fnu_rne_rounding(self, device):
        """
        For every possible e8m0 exponent (256 options) and for every possible
        g, r, s bits of the float32 mantissa, verify that RNE rounding is
        correctly applied when casting from float32 to e8m0

        Note: this code is morally similar to `test_cast_round_trip`, but
        IMO simpler to special case e8m0 here.
        """

        for biased_exponent in range(256):
            # iterate through all the possible options of guard, round, sticky bits
            # for the current exponent
            for grs in range(8):
                # create a positive floating point number with the specified exponent
                # and mantissa guard, round, sticky bits
                uint32_t_start = (biased_exponent << 23) + (grs << 20)
                fp32_start = _int_bits_to_float(uint32_t_start)

                # create an RNE rounded version of the exponent
                if biased_exponent == 255:
                    new_biased_exponent = biased_exponent
                else:
                    lsb = biased_exponent > 0
                    g = grs >> 2
                    r = (grs >> 1) & 0b1
                    s = grs & 0b1
                    new_biased_exponent = _round_e8m0_rne(biased_exponent, lsb, g, r, s)

                # create an RNE rounded version of the float
                fp32_e8m0_fp32_emulated = _int_bits_to_float(new_biased_exponent << 23)

                # now, do the same in PyTorch and see if results match
                fp32_pt_start = torch.full(
                    (1,), fp32_start, device=device, dtype=torch.float
                )
                fp32_pt_e8m0 = fp32_pt_start.to(torch.float8_e8m0fnu)
                fp32_pt_e8m0_fp32 = fp32_pt_e8m0.to(torch.float)

                expected = fp32_e8m0_fp32_emulated
                if biased_exponent == 254 and grs >= 4:
                    # special case rounding up from the largest representable float32 exponent, which
                    # saturates to nan
                    expected = float("nan")
                elif biased_exponent == 255:
                    # special case inf and nan, which becomes nan
                    expected = float("nan")

                actual = fp32_pt_e8m0_fp32.item()

                self.assertEqual(
                    expected,
                    actual,
                    lambda msg: f"{msg}\nexpected: {expected}, actual: {actual}",
                )

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_special_numbers(self, dtype, device):
        """Test special numbers."""

        def compare_binary_with_decimal(binary, decimal, number_name, dtype, device):
            bits_int = int(binary, 2)
            tensor_int = torch.tensor([bits_int], dtype=torch.uint8, device=device)
            tensor_fp8 = tensor_int.view(dtype)
            if number_name == "nan":
                if not tensor_fp8.isnan():
                    raise AssertionError(
                        f"Expected NaN for {number_name}, got {tensor_fp8}"
                    )
            else:
                tensor_fp32 = tensor_fp8.float()
                ref_tensor_fp32 = torch.tensor(
                    [decimal], dtype=torch.float, device=device
                )
                self.assertEqual(tensor_fp32, ref_tensor_fp32, atol=0, rtol=0)

        for number in SPECIAL_NUMBERS[dtype]:
            compare_binary_with_decimal(*number, dtype, device)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_type_promotion_fails(self, dtype, device):
        """Test that float8 is not promoted to higher precision Float Type."""
        for other_dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ]:
            if other_dtype == torch.float64 and self.device_type == "mps":
                continue  # MPS does not support float64
            x = torch.randn(8, device=device).to(dtype)
            y = torch.randn(8, device=device).to(other_dtype)
            with self.assertRaisesRegex(
                RuntimeError, "Promotion for Float8 Types is not supported"
            ):
                x + y

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_empty(self, dtype, device):
        with DeterministicGuard(torch.are_deterministic_algorithms_enabled()):
            for use_deterministic in (True, False):
                torch.use_deterministic_algorithms(use_deterministic)
                torch.empty(4, 4, device=device, dtype=dtype)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_to_string(self, dtype, device):
        x = torch.empty(4, 4, device=device, dtype=dtype)
        str(x)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_finfo(self, dtype, device):
        torch.finfo(dtype)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_cat(self, dtype, device):
        x1 = torch.empty(4, 4, device=device, dtype=dtype)
        x2 = torch.empty(4, 4, device=device, dtype=dtype)
        torch.cat([x1, x2])

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfXPU(*XPU_FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @dtypesIfMPS(*MPS_FLOAT8_DTYPES)
    def test_save_load(self, dtype, device):
        x1 = torch.randint(0, 10, (4, 4), device=device, dtype=torch.uint8).view(dtype)
        with TemporaryFileName() as fname:
            torch.save(x1, fname)
            x1_save_load = torch.load(fname)
            torch.testing.assert_close(x1, x1_save_load, atol=0, rtol=0)


class TestFloat4Dtype(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    # TODO(#146647): make the testing generic for shell dtypes
    def test_float4_e2m1fn_x2(self, device):
        # can create a tensor of dtype float4
        x1 = torch.empty(4096, 4096, device=device, dtype=torch.float4_e2m1fn_x2)

        # can create a string (so printing will work)
        str(x1)

        # can view float4_e2m1fn_x2 as uint8
        x2 = x1.view(torch.uint8)

        # can view uint8 as float4_e2m1fn_x2
        x2.view(torch.float4_e2m1fn_x2)

        # can do equality comparisons
        x3 = copy.deepcopy(x1)
        self.assertEqual(x1, x3, atol=0, rtol=0)

        # can call contiguous on a dim1 slice (calls `copy_` under the hood)
        x1[:, 0:2048].contiguous()

    def test_cast_raises_on_host(self, device):
        # Casting to or from fp4x2 used to device-assert on CUDA, poisoning the
        # context; it must raise on the host instead.
        x_f4 = torch.zeros(4, 8, dtype=torch.uint8, device=device).view(
            torch.float4_e2m1fn_x2
        )
        msg = r"view\(torch.uint8\) and unpack"
        for dtype in (torch.float32, torch.bfloat16, torch.uint8, torch.float8_e4m3fn):
            with self.assertRaisesRegex(NotImplementedError, msg):
                x_f4.to(dtype)
        with self.assertRaisesRegex(NotImplementedError, msg):
            torch.empty(4, 8, device=device).copy_(x_f4)
        with self.assertRaisesRegex(NotImplementedError, msg):
            torch.zeros(4, 8, device=device).to(torch.float4_e2m1fn_x2)
        # zero-element casts are rejected too, rather than silently succeeding
        with self.assertRaisesRegex(NotImplementedError, msg):
            torch.empty(0, dtype=torch.float4_e2m1fn_x2, device=device).float()
        # the device is still usable afterwards
        self.assertEqual((torch.ones(3, device=device) + 1).sum().item(), 6)

    def test_cast_raises_when_tracing(self, device):
        # Meta, FakeTensor, the prims decomposition, and Dynamo tracing must
        # reject the same casts as eager instead of tracing them.
        msg = r"view\(torch.uint8\) and unpack"
        x_meta = torch.empty(4, 8, dtype=torch.float4_e2m1fn_x2, device="meta")
        with self.assertRaisesRegex(NotImplementedError, msg):
            x_meta.to(torch.float32)
        with FakeTensorMode():
            x_fake = torch.empty(4, 8, dtype=torch.float4_e2m1fn_x2, device=device)
            with self.assertRaisesRegex(NotImplementedError, msg):
                x_fake.to(torch.float32)
            with self.assertRaisesRegex(NotImplementedError, msg):
                torch.empty(4, 8, device=device).copy_(x_fake)
            with self.assertRaisesRegex(NotImplementedError, msg):
                torch.ops.prims.convert_element_type(x_fake, torch.float32)
        x = torch.zeros(4, 8, dtype=torch.uint8, device=device).view(
            torch.float4_e2m1fn_x2
        )
        torch._dynamo.reset()
        # fullgraph, so a graph break cannot fall back to the eager check
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            torch.compile(
                lambda t: t.to(torch.float32), backend="eager", fullgraph=True
            )(x)

    @onlyCPU
    def test_shell_dtype_lists_match(self, device):
        # The prims check mirrors the C++ copy_ check; keep their dtype sets equal.
        quantized = {
            torch.qint8,
            torch.qint32,
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
        }
        dtypes = {d for d in vars(torch).values() if isinstance(d, torch.dtype)}
        for dtype in dtypes - quantized - {torch.float32}:
            meta = torch.empty(2, dtype=dtype, device="meta")
            meta_f32 = torch.empty(2, device="meta")
            for cast in (lambda: meta.to(torch.float32), lambda: meta_f32.to(dtype)):
                try:
                    cast()
                    rejected = False
                except NotImplementedError:
                    rejected = True
                self.assertEqual(
                    rejected, dtype in _uncastable_shell_dtypes, msg=str(dtype)
                )

    def test_f4_save_load(self, device):
        x1 = torch.randint(0, 10, (4, 4), device=device, dtype=torch.uint8).view(
            torch.float4_e2m1fn_x2
        )
        with TemporaryFileName() as fname:
            torch.save(x1, fname)
            x1_save_load = torch.load(fname)
            # TODO(#146647): make this and all other shell dtypes support equality
            # comparison
            torch.testing.assert_close(
                x1.view(torch.uint8), x1_save_load.view(torch.uint8), atol=0, rtol=0
            )


def _e5m3fnu_decode_lut():
    """float64 values of all 256 e5m3fnu codes, from the format definition."""
    values = []
    for code in range(256):
        exponent, mantissa = code >> 3, code & 0b111
        if code == 0xFF:
            values.append(float("nan"))
        elif exponent == 0:
            values.append(math.ldexp(mantissa / 8, -14))
        else:
            values.append(math.ldexp(1 + mantissa / 8, exponent - 15))
    return torch.tensor(values, dtype=torch.float64)


def _e5m3fnu_encode_reference(x):
    """Round-to-nearest-even encode of float64 `|x|` by searching the decode LUT:
    the sign is ignored, NaN maps to NaN, values above the max saturate."""
    x = x.abs()
    lut = _e5m3fnu_decode_lut()[:255]
    dist = (x.unsqueeze(-1) - lut).abs()
    min_dist = dist.min(dim=-1, keepdim=True).values
    candidates = (dist == min_dist).nonzero()
    codes = torch.full(x.shape, -1, dtype=torch.int64)
    for row, code in candidates.tolist():
        # ties go to the even code, i.e. the even mantissa
        if codes[row] == -1 or code % 2 == 0:
            codes[row] = code
    codes[x > lut[-1]] = 0xFE
    codes[x.isnan()] = 0xFF
    return codes.to(torch.uint8)


class TestFloat8E5M3FNU(TestCase):
    dtype = torch.float8_e5m3fnu

    def test_decode_all_codes(self, device):
        codes = torch.arange(256, dtype=torch.uint8, device=device)
        actual = codes.view(self.dtype).to(torch.float64)
        self.assertEqual(actual.cpu(), _e5m3fnu_decode_lut(), atol=0, rtol=0)
        self.assertTrue(codes.view(self.dtype).isnan()[0xFF])
        self.assertEqual(codes.view(self.dtype).isnan().sum().item(), 1)

    @parametrize("src_dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_encode(self, device, src_dtype):
        lut = _e5m3fnu_decode_lut()
        finite = lut[:255]
        mids = (finite[:-1] + finite[1:]) / 2
        x = torch.cat(
            (
                finite,
                mids,
                torch.tensor(
                    [
                        2**-18,  # tie between zero and the min subnormal
                        2**-19,
                        2**-30,
                        2**-130,  # float32 subnormal
                        118784.0,  # tie between max (even) and the next code
                        114689.0,
                        118785.0,
                        2.0**100,
                        float("inf"),
                        -0.0,
                        -(2**-17),
                        -1.0,
                        -(2.0**100),
                        float("-inf"),
                        float("nan"),
                    ],
                    dtype=torch.float64,
                ),
            )
        ).to(torch.float32)
        # also cover the float32 neighbors of every code and tie
        x = torch.cat(
            (
                x,
                x.nextafter(torch.tensor(-math.inf)),
                x.nextafter(torch.tensor(math.inf)),
            )
        )
        x = x.to(src_dtype)
        expected = _e5m3fnu_encode_reference(x.to(torch.float64))
        actual = x.to(device).to(self.dtype).view(torch.uint8).cpu()
        self.assertEqual(actual, expected, atol=0, rtol=0)

    def test_shell_ops(self, device):
        codes = torch.arange(256, dtype=torch.uint8, device=device)
        x = codes.view(self.dtype)
        self.assertEqual(x.dtype, self.dtype)
        self.assertEqual(x.view(torch.uint8), codes, atol=0, rtol=0)

        def bits(t):
            return t.view(torch.uint8).cpu()

        # copies across devices and layouts preserve the bits
        self.assertEqual(bits(x.cpu().to(device)), codes.cpu(), atol=0, rtol=0)
        self.assertEqual(bits(x.clone()), codes.cpu(), atol=0, rtol=0)
        self.assertEqual(
            bits(x.view(16, 16).t().contiguous()),
            codes.cpu().view(16, 16).t(),
            atol=0,
            rtol=0,
        )
        y = torch.empty(256, dtype=self.dtype, device=device)
        y.copy_(x)
        self.assertEqual(bits(y), codes.cpu(), atol=0, rtol=0)

        # indexing
        idx = torch.tensor([0x78, 0x00, 0xFE, 0xFF], device=device)
        self.assertEqual(bits(x[idx]), idx.cpu().to(torch.uint8), atol=0, rtol=0)
        self.assertEqual(bits(x[1:4]), codes.cpu()[1:4], atol=0, rtol=0)
        self.assertEqual(
            bits(x.index_select(0, idx)), idx.cpu().to(torch.uint8), atol=0, rtol=0
        )

        # creation and fill
        self.assertEqual(bits(torch.zeros(3, dtype=self.dtype, device=device)).sum(), 0)
        full = torch.full((3,), 1.0, dtype=self.dtype, device=device)
        self.assertEqual(bits(full), torch.full((3,), 0x78, dtype=torch.uint8))
        full.fill_(114688.0)
        self.assertEqual(bits(full), torch.full((3,), 0xFE, dtype=torch.uint8))
        # like the other float8 dtypes, the checked Scalar conversion in
        # fill_kernel rejects values below finfo.min (0) instead of following
        # the cast rule (|x|); some fill paths (one-element CPU tensors,
        # integer Scalars) skip that check, as they do for float8_e8m0fnu
        with self.assertRaisesRegex(RuntimeError, "without overflow"):
            full.fill_(-1.0)
        t = torch.tensor([0.5, 2.0], dtype=self.dtype, device=device)
        self.assertEqual(bits(t), torch.tensor([0x70, 0x80], dtype=torch.uint8))
        self.assertEqual(t.tolist(), [0.5, 2.0])
        self.assertEqual(t[1].item(), 2.0)

        # equality comparison and printing
        self.assertEqual(x, x.clone(), atol=0, rtol=0)
        self.assertIn("dtype=torch.float8_e5m3fnu", str(t))
        self.assertIn("0.5000", str(t))

    def test_finfo(self, device):
        finfo = torch.finfo(self.dtype)
        self.assertEqual(finfo.bits, 8)
        self.assertEqual(finfo.max, 114688.0)
        self.assertEqual(finfo.min, 0.0)
        self.assertEqual(finfo.smallest_normal, 2**-14)
        self.assertEqual(finfo.eps, 2**-3)
        self.assertFalse(self.dtype.is_signed)
        self.assertTrue(self.dtype.is_floating_point)
        self.assertEqual(self.dtype.itemsize, 1)

    def test_promotion_and_dlpack_fail(self, device):
        x = torch.ones(2, dtype=self.dtype, device=device)
        with self.assertRaisesRegex(
            RuntimeError, "Promotion for Float8 Types is not supported"
        ):
            torch.promote_types(self.dtype, torch.float32)
        with self.assertRaisesRegex(BufferError, "not supported by dlpack"):
            torch.utils.dlpack.to_dlpack(x)


instantiate_device_type_tests(
    TestFloat8Dtype, globals(), allow_mps=True, allow_xpu=True
)
instantiate_device_type_tests(TestFloat8E5M3FNU, globals(), except_for="mps")
instantiate_device_type_tests(TestFloat4Dtype, globals(), allow_xpu=True)


def _e2m3_decode_lut():
    """float64 values of all 64 E2M3 codes, from OCP MX v1.0 Section 5.3."""
    values = []
    for code in range(64):
        sign = -1.0 if code & 0b100000 else 1.0
        exponent, mantissa = (code >> 3) & 0b11, code & 0b111
        if exponent == 0:
            values.append(sign * mantissa / 8)
        else:
            values.append(sign * math.ldexp(1 + mantissa / 8, exponent - 1))
    return torch.tensor(values, dtype=torch.float64)


def _as_bytes(x):
    """Each 3-byte element as a trailing dim of 3 uint8 values (any strides)."""
    return x.unsqueeze(-1).view(torch.uint8)


class TestFloat6E2M3FNx4(TestCase):
    dtype = torch.float6_e2m3fn_x4

    def _random(self, *shape, device):
        raw = torch.randint(0, 256, (*shape[:-1], 3 * shape[-1]), dtype=torch.uint8)
        return raw.to(device).view(self.dtype)

    def test_packing_is_little_endian_bitstream(self, device):
        codes = torch.tensor([1, 2, 3, 4], dtype=torch.uint8, device=device)
        self.assertEqual(pack_uint6(codes).tolist(), [0x81, 0x30, 0x10])
        # one value per position, so every byte-boundary crossing is pinned
        for i, expected in enumerate(
            ([0x3F, 0, 0], [0xC0, 0x0F, 0], [0, 0xF0, 0x03], [0, 0, 0xFC])
        ):
            one_hot = torch.zeros(4, dtype=torch.uint8, device=device)
            one_hot[i] = 0x3F
            self.assertEqual(pack_uint6(one_hot).tolist(), expected)
        # the whole buffer is one little-endian bitstream: value i is at bits
        # [6i, 6i + 6), the DLPack sub-byte rule
        codes = torch.randint(0, 64, (8, 64), dtype=torch.uint8, device=device)
        packed = pack_uint6(codes)
        for row, packed_row in zip(codes.tolist(), packed.tolist()):
            stream = int.from_bytes(bytes(packed_row), "little")
            self.assertEqual([(stream >> (6 * i)) & 0x3F for i in range(64)], row)
        self.assertEqual(unpack_uint6(packed), codes)

    def test_decode_all_codes(self, device):
        codes = torch.arange(64, dtype=torch.uint8, device=device)
        x = pack_uint6(codes).view(self.dtype)
        self.assertEqual(x.shape, (16,))
        decoded = _float6_e2m3fn_x4_to_float32(x)
        self.assertEqual(decoded.double().cpu(), _e2m3_decode_lut(), atol=0, rtol=0)

    def test_encode(self, device):
        lut = _e2m3_decode_lut().float()
        x = lut.to(device).reshape(2, 32)
        x_fp6 = _float32_to_float6_e2m3fn_x4(x)
        self.assertEqual(x_fp6.shape, (2, 8))
        self.assertEqual(_float6_e2m3fn_x4_to_float32(x_fp6), x, atol=0, rtol=0)
        # saturates to +-7.5 and rounds ties to even (0.0625 is between 0 and 0.125)
        x = torch.tensor([100.0, -100.0, 0.0625, 0.1875], device=device)
        self.assertEqual(
            _float6_e2m3fn_x4_to_float32(_float32_to_float6_e2m3fn_x4(x)),
            torch.tensor([7.5, -7.5, 0.0, 0.25], device=device),
        )

    def test_metadata(self, device):
        x = torch.empty(4, 8, dtype=self.dtype, device=device)
        self.assertEqual(self.dtype.itemsize, 3)
        self.assertEqual(x.element_size(), 3)
        self.assertEqual(x.nbytes, 4 * 8 * 3)
        self.assertTrue(self.dtype.is_floating_point)
        self.assertEqual(self.dtype.abbr, "f6e2m3fnx4")
        self.assertIn("dtype=torch.float6_e2m3fn_x4", str(x))
        self.assertIn(
            "[129,  48,  16]",
            str(pack_uint6(torch.tensor([1, 2, 3, 4])).view(self.dtype)),
        )

    def test_view_dtype(self, device):
        x = self._random(4, 8, device=device)
        x_u8 = x.view(torch.uint8)
        self.assertEqual(x_u8.shape, (4, 24))
        self.assertEqual(_as_bytes(x_u8.view(self.dtype)), _as_bytes(x))
        self.assertEqual(x.view(torch.float4_e2m1fn_x2).shape, (4, 24))
        self.assertEqual(x_u8[:, 3:].view(self.dtype).view(torch.uint8), x_u8[:, 3:])
        with self.assertRaisesRegex(RuntimeError, "must be divisible by 3"):
            x_u8[:, 1:].view(self.dtype)
        with self.assertRaisesRegex(RuntimeError, "storage_offset"):
            x_u8.flatten()[1:4].view(self.dtype)
        for other in (torch.int16, torch.float32, torch.float64):
            with self.assertRaisesRegex(
                RuntimeError, "one element size must divide the other"
            ):
                x.view(other)
            with self.assertRaisesRegex(
                RuntimeError, "one element size must divide the other"
            ):
                torch.zeros(4, 24, dtype=other, device=device).view(self.dtype)

    def test_byte_level_ops(self, device):
        x = self._random(6, 16, device=device)
        y = self._random(6, 16, device=device)
        xb, yb = _as_bytes(x), _as_bytes(y)
        # each op must also apply to the bytes view, which has a trailing dim of 3
        ops = [
            lambda t: t[:, 8:16],  # second MX block of each row
            lambda t: t[1::2, 3:11],
            lambda t: t.transpose(0, 1).contiguous(),
            lambda t: t.transpose(0, 1).clone(),
            lambda t: t.reshape(3, 32, *t.shape[2:]),
            lambda t: t.unsqueeze(0).expand(2, *t.shape).contiguous(),
        ]
        for op in ops:
            self.assertEqual(_as_bytes(op(x)), op(xb))
        self.assertEqual(_as_bytes(copy.deepcopy(x)), xb)
        for dim in (0, 1):
            self.assertEqual(
                _as_bytes(torch.cat([x, y], dim=dim)), torch.cat([xb, yb], dim=dim)
            )
            self.assertEqual(
                _as_bytes(torch.cat([x, y.t().contiguous().t()], dim=dim)),
                torch.cat([xb, yb], dim=dim),
            )
        self.assertEqual(_as_bytes(torch.stack([x, y])), torch.stack([xb, yb]))
        # large enough for the parallel (non-serial) cat paths
        big = self._random(512, 64, device=device)
        self.assertEqual(
            _as_bytes(torch.cat([big, big], dim=1)),
            torch.cat([_as_bytes(big)] * 2, dim=1),
        )
        out = torch.empty(16, 6, dtype=self.dtype, device=device)
        out.copy_(x.t())
        self.assertEqual(_as_bytes(out), xb.transpose(0, 1))
        # copy_ broadcasts like any other dtype, and errors report the real shapes
        out = torch.empty(6, 16, dtype=self.dtype, device=device)
        out.copy_(x[0])
        self.assertEqual(_as_bytes(out), xb[0].expand(6, 16, 3))
        with self.assertRaisesRegex(RuntimeError, r"output with shape \[16\]"):
            torch.empty(16, dtype=self.dtype, device=device).copy_(x)
        # 0-dim tensors
        scalar = x[1, 2]
        self.assertEqual(_as_bytes(scalar.clone()), xb[1, 2])
        scalar.zero_()
        self.assertEqual(_as_bytes(x[1, 2]).tolist(), [0, 0, 0])

    def test_zeros(self, device):
        # large enough to skip the CPU memset fast path
        for shape in ((4, 8), (1024, 1024)):
            x = torch.zeros(shape, dtype=self.dtype, device=device)
            self.assertEqual(x.view(torch.uint8).count_nonzero().item(), 0)
        x = self._random(64, 32, device=device)
        x_ref = _as_bytes(x).clone()
        x[:, 8:16].zero_()
        x_ref[:, 8:16] = 0
        self.assertEqual(_as_bytes(x), x_ref)

    @onlyCUDA
    def test_cross_device_copy(self, device):
        x = self._random(8, 16, device="cpu")
        for src in (x, x.t()):
            on_device = src.to(device)
            self.assertEqual(on_device.dtype, self.dtype)
            self.assertEqual(_as_bytes(on_device).cpu(), _as_bytes(src))
            self.assertEqual(_as_bytes(on_device.cpu()), _as_bytes(src))

    def test_cast_raises(self, device):
        msg = r"view\(torch.uint8\) and unpack each 3-byte group"
        x = self._random(4, 8, device=device)
        for dtype in (torch.float32, torch.bfloat16, torch.uint8, torch.float8_e4m3fn):
            with self.assertRaisesRegex(NotImplementedError, msg):
                x.to(dtype)
            with self.assertRaisesRegex(NotImplementedError, msg):
                torch.zeros(4, 8, dtype=dtype, device=device).to(self.dtype)
        with self.assertRaisesRegex(NotImplementedError, msg):
            torch.empty(0, dtype=self.dtype, device=device).float()
        with FakeTensorMode():
            x_fake = torch.empty(4, 8, dtype=self.dtype, device=device)
            with self.assertRaisesRegex(NotImplementedError, msg):
                x_fake.to(torch.float32)
            with self.assertRaisesRegex(NotImplementedError, msg):
                torch.ops.prims.convert_element_type(x_fake, torch.float32)
        torch._dynamo.reset()
        with self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg):
            torch.compile(
                lambda t: t.to(torch.float32), backend="eager", fullgraph=True
            )(x)
        # the casts raise on the host, so the device is still usable
        self.assertEqual((torch.ones(3, device=device) + 1).sum().item(), 6)

    def test_compile_views(self, device):
        x = self._random(4, 16, device=device)
        torch._dynamo.reset()
        fn = torch.compile(
            lambda t: t[:, 8:].contiguous().view(torch.uint8),
            backend="aot_eager",
            fullgraph=True,
        )
        self.assertEqual(fn(x), x[:, 8:].contiguous().view(torch.uint8))

    def test_save_load(self, device):
        x = self._random(4, 8, device=device)[:, 2:6]
        with TemporaryFileName() as fname:
            torch.save(x, fname)
            loaded = torch.load(fname)
        self.assertEqual(loaded.dtype, self.dtype)
        self.assertEqual(loaded.shape, x.shape)
        self.assertEqual(_as_bytes(loaded), _as_bytes(x))

    def test_dlpack(self, device):
        x = self._random(4, 8, device=device)
        y = torch.from_dlpack(torch.utils.dlpack.to_dlpack(x))
        self.assertEqual(y.dtype, self.dtype)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.data_ptr(), x.data_ptr())
        z = torch.from_dlpack(x[:, 1:5])
        self.assertEqual(_as_bytes(z), _as_bytes(x[:, 1:5]))


instantiate_device_type_tests(TestFloat6E2M3FNx4, globals(), except_for="mps")


class TestFloat8DtypeCPUOnly(TestCase):
    """
    Test of mul implementation

    NOTE: this is CPU-only for now because adding it to CUDA requires adding yet
    another C++ dtype macro, and there is no use case yet for unscaled float8
    multiplication - doesn't seem worth it.
    """

    hw_classification = HardwareClassification.CPU

    @dtypes(*CUDA_FLOAT8_DTYPES)
    def test_mul(self, dtype):
        # TODO(#113663): remove arithmetic support from all float8 dtypes
        if dtype in (torch.float8_e8m0fnu, torch.float8_e5m3fnu):
            self.skipTest(f"arithmetic not supported for {dtype}")
        shape = (10, 10)
        a = torch.randn(shape)
        a8_simulated = simulate_fp8_precision(a, dtype)
        a8 = a.to(dtype)
        b = torch.randn(shape)
        b8_simulated = simulate_fp8_precision(b, dtype)
        b8 = b.to(dtype)
        mul8 = a8 * b8
        mul8_simulated = (a8_simulated * b8_simulated).to(dtype)
        self.assertEqual(mul8, mul8_simulated)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on Windows yet")
    @dtypes(*CUDA_FLOAT8_DTYPES)
    def test_pt2_traceable_aot_eager(self, dtype):
        if dtype in (torch.float8_e8m0fnu, torch.float8_e5m3fnu):
            self.skipTest(f"PT2 support for {dtype} is not implemented yet")

        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x):
            x = x.to(dtype)
            x = x.float()
            return x

        x = torch.randn(1).requires_grad_()
        f(x).sum().backward()


instantiate_device_type_tests(TestFloat8DtypeCPUOnly, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
