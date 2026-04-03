# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: pt2"]

import torch
from torch._inductor.utils import _type_of


class TestTritonFp8Types:
    """Test that fp8 dtypes are correctly mapped to Triton types."""

    def test_float8_e4m3fn_mapping(self):
        """Test float8_e4m3fn maps to fp8e4nv."""
        result = _type_of(torch.float8_e4m3fn)
        assert result == "fp8e4nv", f"Expected fp8e4nv, got {result}"

    def test_float8_e5m2_mapping(self):
        """Test float8_e5m2 maps to fp8e5."""
        result = _type_of(torch.float8_e5m2)
        assert result == "fp8e5", f"Expected fp8e5, got {result}"

    def test_float8_e4m3fnuz_mapping(self):
        """Test float8_e4m3fnuz maps to fp8e4b8."""
        result = _type_of(torch.float8_e4m3fnuz)
        assert result == "fp8e4b8", f"Expected fp8e4b8, got {result}"

    def test_float8_e5m2fnuz_mapping(self):
        """Test float8_e5m2fnuz maps to fp8e5b16."""
        result = _type_of(torch.float8_e5m2fnuz)
        assert result == "fp8e5b16", f"Expected fp8e5b16, got {result}"

    def test_all_fp8_types(self):
        """Test all fp8 dtype variants map correctly."""
        fp8_types = {
            torch.float8_e4m3fn: "fp8e4nv",
            torch.float8_e5m2: "fp8e5",
            torch.float8_e4m3fnuz: "fp8e4b8",
            torch.float8_e5m2fnuz: "fp8e5b16",
        }
        for dtype, expected in fp8_types.items():
            result = _type_of(dtype)
            assert result == expected, f"dtype {dtype} expected {expected}, got {result}"


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests("test_triton_fp8_types")
