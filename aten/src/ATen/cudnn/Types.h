#pragma once

#include <ATen/Tensor.h>
#include <ATen/cudnn/cudnn-wrapper.h>

namespace at {
namespace native {

TORCH_CUDA_CPP_API cudnnDataType_t
getCudnnDataTypeFromScalarType(const at::ScalarType dtype);
cudnnDataType_t getCudnnDataType(const at::Tensor& tensor);

int64_t cudnn_version();

} // namespace native
} // namespace at
