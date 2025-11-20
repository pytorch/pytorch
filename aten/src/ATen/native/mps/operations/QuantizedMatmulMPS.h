#pragma once

#include <ATen/native/mps/OperationUtils.h>

namespace at {
namespace native {
TORCH_API at::Tensor _quantized_matmul_bf16i4bf16_rowwise_mps(
    const at::Tensor& self,
    const at::Tensor& weight,
    const at::Tensor& scales,
    const at::Tensor& zeros);
} // namespace native
} // namespace at
