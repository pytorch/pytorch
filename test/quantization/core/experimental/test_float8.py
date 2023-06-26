# Owner(s): ["oncall: quantization"]

import torch
import pytest

MASK_FLOAT32_152 = torch.tensor(2145386496, dtype=torch.int) # 0 11111111 11000000000000000000000b
MASK_FLOAT32_143 = torch.tensor(2146435072, dtype=torch.int) # 0 11111111 11100000000000000000000b
MASK_FLOAT32 = {torch.float8_e5m2: MASK_FLOAT32_152, torch.float8_e4m3: MASK_FLOAT32_143}

MASK_ROUND_FLOAT32_152 = torch.tensor(1048575, dtype=torch.int) # 0 00000000 00011111111111111111111b
MASK_ROUND_FLOAT32_143 = torch.tensor(524287, dtype=torch.int) # 0 00000000 00001111111111111111111b
MASK_ROUND_FLOAT32 = {torch.float8_e5m2: MASK_ROUND_FLOAT32_152, torch.float8_e4m3: MASK_ROUND_FLOAT32_143}

MASK_BFLOAT16_152 = torch.tensor(32736, dtype=torch.short) # 0 11111111 1100000b
MASK_BFLOAT16_143 = torch.tensor(32752, dtype=torch.short) # 0 11111111 1110000b
MASK_BFLOAT16 = {torch.float8_e5m2: MASK_BFLOAT16_152, torch.float8_e4m3: MASK_BFLOAT16_143}

MASK_ROUND_BFLOAT16_152 = torch.tensor(15, dtype=torch.short) # 0 00000000 0001111b
MASK_ROUND_BFLOAT16_143 = torch.tensor(7, dtype=torch.short) # 0 00000000 0000111b
MASK_ROUND_BFLOAT16 = {torch.float8_e5m2: MASK_ROUND_BFLOAT16_152, torch.float8_e4m3: MASK_ROUND_BFLOAT16_143}

FP8_MAX_152 = torch.tensor(57344, dtype=torch.float)
FP8_MAX_143 = torch.tensor(448, dtype=torch.float)
FP8_MAX = {torch.float8_e5m2: FP8_MAX_152, torch.float8_e4m3: FP8_MAX_143}

def simulateFp8Precision(input, variant):
    dtype = input.dtype
    if dtype == torch.float:
        int_type = torch.int
        mask = MASK_FLOAT32[variant]
        mask_round = MASK_ROUND_FLOAT32[variant]
        excessive_bits = torch.tensor(21, dtype=int_type)
    else:
        int_type = torch.short
        mask = MASK_BFLOAT16[variant]
        mask_round = MASK_ROUND_BFLOAT16[variant]
        excessive_bits = torch.tensor(5, dtype=int_type)
    signs = torch.where(input < 0.0, -1.0, 1.0).to(dtype)
    asInt = input.view(int_type)
    mant_odd = torch.bitwise_and(torch.bitwise_right_shift(asInt, excessive_bits), torch.tensor(1, dtype=int_type))
    asInt_masked = asInt + mask_round
    asInt_odded = asInt_masked + mant_odd
    masked = torch.bitwise_and(asInt_odded, mask)
    return masked.view(dtype)*signs

@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3])
def test_creation_with_zeros(dtype):
    x = torch.zeros(8, dtype=torch.float)
    x8 = torch.zeros(8, dtype=dtype)
    assert torch.count_nonzero(x == x8) == x.numel()

@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3])
def test_cast_to_float8(dtype):
    x = torch.rand((100, 100))*FP8_MAX[dtype]
    x = torch.cat((x, -x))
    x8 = x.to(dtype)
    x8_simulated = simulateFp8Precision(x, dtype)
    assert torch.count_nonzero(x8_simulated == x8) == x.numel()

@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3])
def test_mul(dtype):
    a = torch.randn((10, 10))
    a8_simulated = simulateFp8Precision(a, dtype)
    a8 = a.to(dtype)
    b = torch.randn((10, 10))
    b8_simulated = simulateFp8Precision(b, dtype)
    b8 = b.to(dtype)
    mul8 = a8 * b8
    mul8_simulated = (a8_simulated * b8_simulated).to(dtype)
    assert torch.count_nonzero(mul8 == mul8_simulated) == a.numel()

def compare_binary_with_decimal(binary, decimal, number_name, dtype):
    bits_int = int(binary, 2)
    tensor_int = torch.tensor([bits_int], dtype=torch.uint8)
    tensor_fp8 = tensor_int.view(dtype)
    if number_name == "nan":
        assert tensor_fp8.isnan()
    else:
        tensor_fp32 = tensor_fp8.float()
        ref_tensor_fp32 = torch.tensor([decimal], dtype=torch.float)
        assert torch.allclose(tensor_fp32, ref_tensor_fp32), f"{number_name} failed: expected {decimal}, got {tensor_fp32.item()}"

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
        ("00000100", 2 ** -14, "min_normal"),
        ("10000100", -1 * (2 ** -14), "neg_min_normal"),
        ("00000011", 0.75 * (2 ** -14), "max_subnorm"),
        ("10000011", -0.75 * (2 ** -14), "neg_max_subnorm"),
        ("00000001", 2 ** -16, "min_subnorm"),
        ("10000001", -1 * (2 ** -16), "neg_min_subnorm")
    ],
    torch.float8_e4m3: [
        ("01111111", float("nan"), "nan"),
        ("11111111", float("nan"), "nan"),
        ("00000000", 0.0, "zero"),
        ("10000000", -0.0, "neg_zero"),
        ("01111110", 448.0, "max_normal"),
        ("11111110", -448.0, "neg_max_normal"),
        ("00001000", 2 ** -6, "min_normal"),
        ("10001000", -1 * (2 ** -6), "neg_min_normal"),
        ("00000111", 0.875 * (2 ** -6), "max_subnorm"),
        ("10000111", -0.875 * (2 ** -6), "neg_max_subnorm"),
        ("00000001", 2 ** -9, "min_subnorm"),
        ("10000001", -1 * (2 ** -9), "neg_min_subnorm")
    ]
}

@pytest.mark.parametrize("dtype", [torch.float8_e5m2, torch.float8_e4m3])
def test_special_numbers(dtype):
    for number in SPECIAL_NUMBERS[dtype]:
        compare_binary_with_decimal(*number, dtype)

if __name__ == '__main__':
    pytest.main([__file__])