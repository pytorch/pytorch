#pragma once

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/Tensor.h>

namespace at { namespace native {

TORCH_CUDA_API cudnnDataType_t getCudnnDataTypeFromScalarType(const at::ScalarType dtype);
cudnnDataType_t getCudnnDataType(const at::Tensor& tensor);

int64_t cudnn_version();

}}  // namespace at::cudnn
