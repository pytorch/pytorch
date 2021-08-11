// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once


#include <ATen/ATen.h>

void FloatToBFloat16Quantized_ref(
    const float* const input,
    const size_t nrows,
    const size_t ncols,
    uint16_t* const output);

void BFloat16QuantizedToFloat_ref(
    const at::BFloat16* const input,
    const size_t nrows,
    const size_t ncols,
    float* const output);

at::Tensor _float_to_bfloat16_cpu(const at::Tensor& input);
at::Tensor _bfloat16_to_float_cpu(const at::Tensor& input);
