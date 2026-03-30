# Owner(s): ["oncall: quantization"]

import copy
import struct
import unittest

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    DeterministicGuard,
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
]

CUDA_FLOAT8_DTYPES = [
    torch.float8_e5m2,
    torch.float8_e4m3fn,
    torch.float8_e8m0fnu,
]

# The following information are not yet provided by torch.finfo.

MANTISSA_BITS = {
    torch.float8_e5m2: 2,
    torch.float8_e5m2fnuz: 2,
    torch.float8_e4m3fn: 3,
    torch.float8_e4m3fnuz: 3,
    torch.float8_e8m0fnu: 0,
}

# As in np.finfo(dtype).minexp
MINEXP = {
    torch.float8_e5m2: -14,
    torch.float8_e5m2fnuz: -15,
    torch.float8_e4m3fn: -6,
    torch.float8_e4m3fnuz: -7,
    torch.float8_e8m0fnu: -127,
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
}

FLOAT8_DTYPES_WITH_INF = [torch.float8_e5m2]


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

    # Replace overflows with inf/NaN as appropriate (no saturation)
    have_inf = variant in FLOAT8_DTYPES_WITH_INF
    vals[vals > torch.finfo(variant).max] = torch.inf if have_inf else torch.nan

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
    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
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
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    @parametrize("get_input", ROUND_TRIP_TEST_CASES)
    def test_cast_round_trip(self, dtype, get_input, device):
        """Numerical test of float8 conversion, by performing a round-trip cast
        to the float8 dtype and back to float32, comparing against simulated
        lower precision."""
        if dtype is torch.float8_e8m0fnu:
            return unittest.skip("numerics for e8m0fnu are tested elsewhere")

        x = get_input(dtype, device)
        x = torch.cat((x, -x))
        x8 = x.to(dtype)
        x8_simulated = simulate_fp8_precision(x, dtype)
        self.assertEqual(x8_simulated, x8.float())

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
                    expected, actual, f"expected: {expected}, actual: {actual}"
                )

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
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
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    def test_type_promotion_fails(self, dtype, device):
        """Test that float8 is not promoted to higher precision Float Type."""
        for other_dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ]:
            x = torch.randn(8, device=device).to(dtype)
            y = torch.randn(8, device=device).to(other_dtype)
            with self.assertRaisesRegex(
                RuntimeError, "Promotion for Float8 Types is not supported"
            ):
                x + y

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    def test_empty(self, dtype, device):
        with DeterministicGuard(torch.are_deterministic_algorithms_enabled()):
            for use_deterministic in (True, False):
                torch.use_deterministic_algorithms(use_deterministic)
                torch.empty(4, 4, device=device, dtype=dtype)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    def test_to_string(self, dtype, device):
        x = torch.empty(4, 4, device=device, dtype=dtype)
        str(x)

    @dtypes(*FLOAT8_DTYPES)
    def test_finfo(self, dtype, device):
        torch.finfo(dtype)

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    def test_cat(self, dtype, device):
        x1 = torch.empty(4, 4, device=device, dtype=dtype)
        x2 = torch.empty(4, 4, device=device, dtype=dtype)
        torch.cat([x1, x2])

    @dtypes(*FLOAT8_DTYPES)
    @dtypesIfCUDA(*CUDA_FLOAT8_DTYPES)
    def test_save_load(self, dtype, device):
        x1 = torch.randint(0, 10, (4, 4), device=device, dtype=torch.uint8).view(dtype)
        with TemporaryFileName() as fname:
            torch.save(x1, fname)
            x1_save_load = torch.load(fname)
            torch.testing.assert_close(x1, x1_save_load, atol=0, rtol=0)


class TestFloat4Dtype(TestCase):
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


instantiate_device_type_tests(TestFloat8Dtype, globals())
instantiate_device_type_tests(TestFloat4Dtype, globals())


class TestFloat8DtypeCPUOnly(TestCase):
    """
    Test of mul implementation

    NOTE: this is CPU-only for now because adding it to CUDA requires adding yet
    another C++ dtype macro, and there is no use case yet for unscaled float8
    multiplication - doesn't seem worth it.
    """

    @dtypes(*CUDA_FLOAT8_DTYPES)
    def test_mul(self, dtype):
        # TODO(#113663): remove arithmetic support from all float8 dtypes
        if dtype is torch.float8_e8m0fnu:
            return unittest.skip("arithmetic not supported for torch.float8_e8m0fnu")
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
        if dtype is torch.float8_e8m0fnu:
            return unittest.skip(
                "PT2 support for torch.float8_e8m0fnu is not implemented yet"
            )

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
