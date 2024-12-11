import copy
import io
import itertools
import random
import re
import unittest
import warnings
from typing import List, Tuple

import pytest

import torch
import torch.nn as nn
from torchao.float8.float8_python_api import addmm_float8_unwrapped
from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
    ScaledMMConfig,
)
from torchao.float8.float8_utils import (
    compute_error,
    e4m3_dtype,
    e5m2_dtype,
    fp8_tensor_statistics,
    FP8_TYPES,
    tensor_to_scale,
)
class TestScaledMM:
    @pytest.mark.parametrize(
        "base_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("use_fast_accum", [True, False])
    def test_scaled_mm_vs_emulated(self, base_dtype, use_fast_accum):
        torch.manual_seed(42)
        input_dtype = e4m3_dtype
        output_dtype = base_dtype
        compare_type = torch.float32

        a = torch.randn(16, 16, device="xpu", dtype=base_dtype)
        b = torch.randn(32, 16, device="xpu", dtype=base_dtype).t()

        a_scale = tensor_to_scale(a, input_dtype).float()
        b_scale = tensor_to_scale(b, input_dtype).float()

        a_fp8 = hp_tensor_and_scale_to_float8(a, a_scale, input_dtype)
        b_fp8 = hp_tensor_and_scale_to_float8(b, b_scale, input_dtype)

        out_scaled_mm = addmm_float8_unwrapped(
            a_fp8._data,
            a_fp8._scale,
            b_fp8._data,
            b_fp8._scale,
            output_dtype=output_dtype,
            use_fast_accum=use_fast_accum,
        )
        out_emulated = torch.mm(
            a_fp8._data.float() / a_fp8._scale,
            b_fp8._data.float() / b_fp8._scale,
        ).to(output_dtype)

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_emulated = out_emulated.to(compare_type)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3
        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    def test_different_configs_error(self):
        x_fp32 = torch.randn(16, 16, device="xpu")
        x_scale = torch.tensor(1.0, device="xpu")
        fp8_dtype = e4m3_dtype
        linear_config_a = LinearMMConfig(
            ScaledMMConfig(False, True, False, False),
            ScaledMMConfig(False, False, False, False),
            ScaledMMConfig(False, False, False, False),
        )
        linear_config_b = LinearMMConfig(
            ScaledMMConfig(True, True, False, False),
            ScaledMMConfig(True, False, False, False),
            ScaledMMConfig(True, False, False, False),
        )
        a = hp_tensor_and_scale_to_float8(
            x_fp32,
            x_scale,
            fp8_dtype,
            linear_config_a,
            GemmInputRole.INPUT,
        )
        b = hp_tensor_and_scale_to_float8(
            x_fp32,
            x_scale,
            fp8_dtype,
            linear_config_b,
            GemmInputRole.WEIGHT,
        )
        with pytest.raises(
            AssertionError,
            match="linear_mm_config.output mismatch",
        ):
            a @ b

    @pytest.mark.parametrize(
        "base_dtype", [torch.float16, torch.bfloat16, torch.float32]
    )
    @pytest.mark.parametrize("use_fast_accum", [True, False])
    def test_pad_inner_dim(self, base_dtype, use_fast_accum):
        torch.manual_seed(42)
        input_dtype = e4m3_dtype
        compare_type = torch.float32

        a = torch.randn(16, 41, device="xpu", dtype=base_dtype)
        b = torch.randn(41, 128, device="xpu", dtype=base_dtype)

        a_scale = tensor_to_scale(a, input_dtype).float()
        b_scale = tensor_to_scale(b, input_dtype).float()

        a_fp8 = hp_tensor_and_scale_to_float8(
            a, a_scale, input_dtype, None, GemmInputRole.INPUT
        )
        b_fp8 = hp_tensor_and_scale_to_float8(
            b, b_scale, input_dtype, None, GemmInputRole.WEIGHT
        )

        # with pytest.raises(
        #     RuntimeError,
        #     match=re.escape(
        #         "Expected trailing dimension of mat1 to be divisible by 16 but got mat1 shape: (16x41)."
        #     ),
        # ):
        #     a_fp8 @ b_fp8

        scaled_mm_config = ScaledMMConfig(False, use_fast_accum, False, True)
        pad_config = LinearMMConfig(
            scaled_mm_config, scaled_mm_config, scaled_mm_config
        )

        a_fp8 = hp_tensor_and_scale_to_float8(
            a,
            a_scale,
            input_dtype,
            pad_config,
            GemmInputRole.INPUT,
        )
        b_fp8 = hp_tensor_and_scale_to_float8(
            b,
            b_scale,
            input_dtype,
            pad_config,
            GemmInputRole.WEIGHT,
        )
        out_padded = a_fp8 @ b_fp8
        out_padded.to(compare_type)

        emulated_scaled_mm_config = ScaledMMConfig(True, use_fast_accum, False, False)
        emulated_config = LinearMMConfig(
            emulated_scaled_mm_config,
            emulated_scaled_mm_config,
            emulated_scaled_mm_config,
        )
        a_fp8 = hp_tensor_and_scale_to_float8(
            a,
            a_scale,
            input_dtype,
            emulated_config,
            GemmInputRole.INPUT,
        )
        b_fp8 = hp_tensor_and_scale_to_float8(
            b,
            b_scale,
            input_dtype,
            emulated_config,
            GemmInputRole.WEIGHT,
        )
        out_emualted = a_fp8 @ b_fp8
        out_emualted.to(compare_type)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3
        torch.testing.assert_close(out_padded, out_emualted, atol=atol, rtol=rtol)
