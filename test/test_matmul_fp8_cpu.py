# Owner(s): ["module: mkldnn"]

import unittest
from typing import Optional

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32

e4m3_type = torch.float8_e4m3fn
e5m2_type = torch.float8_e5m2
E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max


@unittest.skipIf(not torch.backends.mkldnn.is_available(), "MKL-DNN build is disabled")
@unittest.skipIf(
    not torch.cpu._is_amx_tile_supported(), "FP8 cannot run on the current CPU platform"
)
class TestFP8MatmulCpu(TestCase):
    def _test_tautological_mm(
        self,
        device: str = "cpu",
        x_dtype: torch.dtype = e4m3_type,
        y_dtype: torch.dtype = e4m3_type,
        out_dtype: Optional[torch.dtype] = None,
        size: int = 16,
    ) -> None:
        x_fp8 = torch.rand(size, size, device=device).to(x_dtype)
        y_fp8 = torch.eye(size, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = torch._scaled_mm(x_fp8, y_fp8, scale_a, scale_b, out_dtype=out_dtype)
        if out_dtype is not None:
            self.assertEqual(out_dtype, out_fp8.dtype)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    def test_float8_basics(self, device) -> None:
        self._test_tautological_mm(device, e4m3_type, e4m3_type, size=16)
        # TODO: The following 2 tests are mixed dtypes between src and weight,
        # which will be enabled in oneDNN v3.6.
        # self._test_tautological_mm(device, e4m3_type, e5m2_type, size=32)
        # self._test_tautological_mm(device, e5m2_type, e4m3_type, size=48)
        self._test_tautological_mm(device, e5m2_type, e5m2_type)

        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)
        with self.assertRaises(AssertionError):
            self._test_tautological_mm(device, out_dtype=e5m2_type)

    def test_float8_scale(self, device) -> None:
        size = (16, 16)
        x = torch.full(size, 0.5, device=device, dtype=e4m3_type)
        # TODO: will use y = torch.full(size, 0.5, device=device, dtype=e5m2_type).t()
        # after upgrading to oneDNN v3.6.
        y = torch.full(size, 0.5, device=device, dtype=e4m3_type).t()
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4.0, device=device))
        out_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)

    def test_float8_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.ones((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), 0.25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.half)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        outb_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, bias=bias)
        # this fails on ROCm currently because hipblaslt doesn't have amax op
        out_fp32 = out_fp8.to(torch.float32)
        outb_fp32 = outb_fp8.to(torch.float32)
        difference = torch.abs(out_fp32 - outb_fp32)
        self.assertEqual(
            difference, torch.tensor(4.0, device=device).expand_as(out_fp32)
        )


instantiate_device_type_tests(TestFP8MatmulCpu, globals(), only_for=("cpu"))

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
