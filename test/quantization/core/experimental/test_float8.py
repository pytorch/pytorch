# Owner(s): ["oncall: quantization"]

import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)

# Masks for float8 simulation

# 0 11111111 11000000000000000000000b
MASK_152 = torch.tensor(2145386496, dtype=torch.int)
# 0 11111111 11100000000000000000000b
MASK_143 = torch.tensor(2146435072, dtype=torch.int)
MASK = {
    torch.float8_e5m2: MASK_152,
    torch.float8_e4m3fn: MASK_143,
}

# 0 00000000 00011111111111111111111b
MASK_ROUND_152 = torch.tensor(1048575, dtype=torch.int)
# 0 00000000 00001111111111111111111b
MASK_ROUND_143 = torch.tensor(524287, dtype=torch.int)
MASK_ROUND = {
    torch.float8_e5m2: MASK_ROUND_152,
    torch.float8_e4m3fn: MASK_ROUND_143,
}

FP8_MAX_152 = torch.tensor(57344, dtype=torch.float)
FP8_MAX_143 = torch.tensor(448, dtype=torch.float)
FP8_MAX = {torch.float8_e5m2: FP8_MAX_152, torch.float8_e4m3fn: FP8_MAX_143}

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
}


def simulateFp8Precision(input, variant):
    dtype = torch.float
    int_type = torch.int
    mask = MASK[variant]
    mask_round = MASK_ROUND[variant]
    excessive_bits = torch.tensor(21, dtype=int_type)

    signs = torch.where(input < 0.0, -1.0, 1.0).to(dtype)
    asInt = torch.bitwise_and(input.view(int_type), 2147483647)

    mant_odd = torch.bitwise_and(
        torch.bitwise_right_shift(asInt, excessive_bits),
        torch.tensor(1, dtype=int_type),
    )
    asInt_masked = asInt + mask_round
    asInt_odded = asInt_masked + mant_odd
    masked = torch.bitwise_and(asInt_odded, mask)
    return masked.view(dtype) * signs


class TestFloat8Dtype(TestCase):
    """
    Sanity test for zeros comparison
    """

    @parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
    def test_creation_with_zeros(self, dtype, device):
        x = torch.zeros(8, dtype=torch.float, device=device)
        x8 = torch.zeros(8, dtype=dtype, device=device)
        self.assertEqual(x, x8.float())

    """
        Numerical test of float8 conversion
    """

    @parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
    def test_cast_to_float8(self, dtype, device):
        x = torch.rand((100, 100), device=device) * FP8_MAX[dtype]
        x = torch.cat((x, -x))
        x8 = x.to(dtype)
        x8_simulated = simulateFp8Precision(x, dtype)
        self.assertEqual(x8_simulated, x8.float())

    """
        Test special numbers
    """

    @parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
    def test_special_numbers(self, dtype, device):
        def compare_binary_with_decimal(binary, decimal, number_name, dtype, device):
            bits_int = int(binary, 2)
            tensor_int = torch.tensor([bits_int], dtype=torch.uint8, device=device)
            tensor_fp8 = tensor_int.view(dtype)
            if number_name == "nan":
                assert tensor_fp8.isnan()
            else:
                tensor_fp32 = tensor_fp8.float()
                ref_tensor_fp32 = torch.tensor(
                    [decimal], dtype=torch.float, device=device
                )
                self.assertEqual(tensor_fp32, ref_tensor_fp32)

        for number in SPECIAL_NUMBERS[dtype]:
            compare_binary_with_decimal(*number, dtype, device)


instantiate_device_type_tests(TestFloat8Dtype, globals())


class TestFloat8DtypeCPUOnly(TestCase):

    """
    Test of mul implementation
    # Note: this is cpu-only for now because adding it to CUDA requires
    adding yet c++ dtype macro, and there is no use case yet for unscaled
    float8 multiplication - doesn't seem worth it.
    """

    @parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
    def test_mul(self, dtype):
        shape = (10, 10)
        a = torch.randn(shape)
        a8_simulated = simulateFp8Precision(a, dtype)
        a8 = a.to(dtype)
        b = torch.randn(shape)
        b8_simulated = simulateFp8Precision(b, dtype)
        b8 = b.to(dtype)
        mul8 = a8 * b8
        mul8_simulated = (a8_simulated * b8_simulated).to(dtype)
        self.assertEqual(mul8, mul8_simulated)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on Windows yet")
    @parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3fn])
    def test_pt2_traceable_aot_eager(self, dtype):
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
