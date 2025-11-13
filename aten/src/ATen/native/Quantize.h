#pragma once
#include <ATen/ATen.h>
#include <tuple>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> quantize_mx_cpu(
    const Tensor& self,
    int64_t block_size,
    at::ScalarType dtype,
    int64_t rounding_mode);


// Meta kernel for torch.compile (shape inference only)
std::tuple<Tensor, Tensor> quantize_mx_meta(
    const Tensor& self,
    int64_t block_size,
    at::ScalarType dtype,
    int64_t rounding_mode);

} // namespace native
} // namespace at