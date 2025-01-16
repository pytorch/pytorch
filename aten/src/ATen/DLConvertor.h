#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/dlpack.h>

// this convertor will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the ATen Tensor

namespace at {

// This trait class is used for retrieving the different PyCapsule names for
// both DLPack tensor classes: `DLManagedTensor` and `DLManagedTensorVersioned`.
//
// Each specialization should contain the following 2 traits:
//   - `capsule`: actual name of the capsule
//   - `used`: name of the capsule after using it
template <class T>
struct DLPackTraits {};

template<>
struct DLPackTraits<DLManagedTensor> {
    inline static const char* capsule = "dltensor";
    inline static const char* used = "used_dltensor";
};

template<>
struct DLPackTraits<DLManagedTensorVersioned> {
    inline static const char* capsule = "dltensor_versioned";
    inline static const char* used = "used_dltensor_versioned";
};

TORCH_API ScalarType toScalarType(const DLDataType& dtype);
TORCH_API DLManagedTensorVersioned* toDLPack(const Tensor& src);
TORCH_API DLManagedTensor* toDLPackUnversioned(const Tensor& src);
TORCH_API Tensor
fromDLPack(DLTensor& dl_tensor, std::function<void(void*)> deleter);
TORCH_API DLDataType getDLDataType(const Tensor& t);
TORCH_API DLDevice getDLContext(const Tensor& tensor, const int64_t& device_id);

} // namespace at
