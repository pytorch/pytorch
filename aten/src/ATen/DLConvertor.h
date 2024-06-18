#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/dlpack.h>

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the ATen Tensor

namespace at {

TORCH_API ScalarType toScalarType(const DLDataType& dtype);
TORCH_API DLManagedTensor* toDLPack(const Tensor& src);
TORCH_API Tensor fromDLPack(DLManagedTensor* src);
C10_DEPRECATED_MESSAGE("Please migrate to a non-const variant")
inline Tensor fromDLPack(const DLManagedTensor* src) {
  return fromDLPack(const_cast<DLManagedTensor*>(src));
}
TORCH_API Tensor
fromDLPack(DLManagedTensor* src, std::function<void(void*)> deleter);
TORCH_API DLDataType getDLDataType(const Tensor& t);
TORCH_API DLDevice getDLContext(const Tensor& tensor, const int64_t& device_id);

} // namespace at
