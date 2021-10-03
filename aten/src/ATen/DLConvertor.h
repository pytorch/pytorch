#pragma once

#include <ATen/Tensor.h>
#include <ATen/ATen.h>
#include <ATen/dlpack.h>

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the ATen Tensor

namespace at {

TORCH_API ScalarType toScalarType(const DLDataType& dtype);
TORCH_API DLManagedTensor* toDLPack(const Tensor& src);
TORCH_API Tensor fromDLPack(const DLManagedTensor* src);
TORCH_API DLDataType getDLDataType(const Tensor& t);
TORCH_API DLContext getDLContext(const Tensor& tensor, const int64_t& device_id);

} //namespace at
