// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once

#include <ATen/ATen.h>
#include <vector>

at::Tensor _float_to_bfloat16_gpu(const at::Tensor& input);
at::Tensor _bfloat16_to_float_gpu(const at::Tensor& input);
