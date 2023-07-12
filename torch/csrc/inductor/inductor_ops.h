#pragma once

#include <ATen/Tensor.h>

namespace torch {
namespace inductor {

TORCH_API at::Tensor _mm_plus_mm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& d,
    at::Tensor& out);

} // namespace inductor
} // namespace torch
