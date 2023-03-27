# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase

# TODO(before land): consider moving to test_torch.py, for better consistency
# with other dtypes
class TestFloat8(TestCase):

    def test_creation_with_zeros(self):
        x = torch.zeros(8, dtype=torch.float8_e4m3fn)

    def test_e4m3fn_casts(self):
        for dtype in (torch.float32, torch.float16):
            x = torch.randn(16, dtype=torch.float)
            x_fp8 = x.to(torch.float8_e4m3fn)
            x_orig_dtype = x_fp8.to(torch.float)

    def test_e4m3fn_numerics(self):
        # ensure that our format matches https://arxiv.org/pdf/2209.05433.pdf, Table 1

        def _compare(bits_str, expected_fp32, comp_name):
            bits_int = int(bits_str, 2)
            tensor_int = torch.tensor([bits_int], dtype=torch.uint8)
            tensor_fp8 = tensor_int.view(torch.float8_e4m3fn)
            tensor_fp32 = tensor_fp8.float()
            ref_tensor_fp32 = torch.tensor([expected_fp32], dtype=torch.float)
            self.assertTrue(
                torch.allclose(tensor_fp32, ref_tensor_fp32),
                f"{comp_name} failed: expected {expected_fp32}, got {tensor_fp32.item()}")

        # TODO(future PR): figure out what to do with infinity, currently it saturates to max_pos
        # _compare("inf", "01111111", float("inf"))
        # _compare("neg_inf", "11111111", -1 * float("inf"))
        _compare("00000000", 0.0, "zero")
        _compare("10000000", -0.0, "neg_zero")
        _compare("01111110", 448.0, "max_normal")
        _compare("11111110", -448.0, "neg_max_normal")
        _compare("00001000", 2 ** -6, "min_normal")
        _compare("10001000", -1 * (2 ** -6), "neg_min_normal")
        _compare("00000111", 0.875 * (2 ** -6), "max_subnorm")
        _compare("10000111", -0.875 * (2 ** -6), "neg_max_subnorm")
        _compare("00000001", 2 ** -9, "min_subnorm")
        _compare("10000001", -1 * (2 ** -9), "neg_min_subnorm")

    # TODO(before land): better verification of casting numerics
    # TODO(future PR): CUDA support


if __name__ == '__main__':
    run_tests()
