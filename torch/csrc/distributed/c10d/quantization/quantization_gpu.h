// (c) Meta Platforms, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree

#pragma once

#include <ATen/ATen.h>
#include <vector>

namespace torch {
namespace distributed {
namespace c10d {
namespace quantization {

at::Tensor _float_to_bfloat16_cuda(const at::Tensor& input);
at::Tensor _bfloat16_to_float_cuda(const at::Tensor& input);

} // namespace quantization
} // namespace c10d
} // namespace distributed
} // namespace torch
