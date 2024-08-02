#pragma once

#include <ATen/Tensor.h>

namespace torch::inductor {

TORCH_API at::Tensor _mm_plus_mm_out(
    at::Tensor& out,
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& d);

// After adding _mm_plus_mm_out, this should not be exposed and called by model
// code. Keeping it around for backward compatibility. Will be deprecated later.
TORCH_API at::Tensor _mm_plus_mm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c,
    const at::Tensor& d,
    at::Tensor& out);

TORCH_API at::Tensor _alloc_from_pool(
    const at::Tensor& self,
    int64_t offset_bytes,
    at::ScalarType dtype,
    at::IntArrayRef size,
    at::IntArrayRef stride);

// Similar to as_strided with the following differences
// - offset is added to the existing offset (rather than replacing it)
// - view tracking is disabled similar to unsafe_view
TORCH_API at::Tensor _reinterpret_tensor(
    const at::Tensor& self,
    at::IntArrayRef size,
    at::IntArrayRef stride,
    int64_t offset_increment = 0);

} // namespace torch::inductor
