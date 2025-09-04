#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/dlpack.h>

// this converter will:
// 1) take a Tensor object and wrap it in the DLPack tensor
// 2) take a dlpack tensor and convert it to the ATen Tensor

namespace at {

TORCH_API ScalarType toScalarType(const DLDataType& dtype);
TORCH_API DLManagedTensor* toDLPack(const Tensor& src);
TORCH_API struct DLManagedTensorVersioned* toDLPackVersioned(const Tensor& src);
TORCH_API Tensor
fromDLPack(DLManagedTensor* src, std::function<void(void*)> deleter = {});
TORCH_API Tensor fromDLPackVersioned(
    DLManagedTensorVersioned* src,
    std::function<void(void*)> deleter = {});
TORCH_API DLDataType getDLDataType(const Tensor& t);
TORCH_API DLDevice getDLContext(const Tensor& tensor, const int64_t& device_id);

// Copies the Tensor if there's a device mismatch or copy is forced.
// This should be used before actually creating the DLPack capsule.
TORCH_API Tensor maybeCopyTensor(
    const Tensor& data,
    std::optional<DLDevice> optional_dl_device,
    std::optional<bool> copy);

// Converts the given at::Device into a DLDevice.
TORCH_API DLDevice torchDeviceToDLDevice(at::Device device);

// This trait class is used for retrieving different attributes, such as the
// PyCapsule names and conversion functions for both DLPack tensor classes:
// `DLManagedTensor` and `DLManagedTensorVersioned`.
//
// Each specialization should contain the following 2 traits:
//   - `capsule`: actual name of the capsule
//   - `used`: name of the capsule after using it
//   - `toDLPack`: function for converting a tensor into a DLPack capsule
//   - `fromDLPack`: function for creating a tensor from a DLPack capsule
//
// While `toDLPack` is the directly exposed to Python, `fromDLPack` is not.
// Although it contains the core implementation, it lacks the required book
// keeping logic contained in its caller `tensor_fromDLPack`.
//
// That said, `fromDLPack` is used directly in a few DLPack tests that live
// inside ATen (no Python available).
template <class T>
struct DLPackTraits {};

template <>
struct DLPackTraits<DLManagedTensor> {
  inline static const char* capsule = "dltensor";
  inline static const char* used = "used_dltensor";
  inline static auto toDLPack = at::toDLPack;
  inline static auto fromDLPack = at::fromDLPack;
};

template <>
struct DLPackTraits<DLManagedTensorVersioned> {
  inline static const char* capsule = "dltensor_versioned";
  inline static const char* used = "used_dltensor_versioned";
  inline static auto toDLPack = at::toDLPackVersioned;
  inline static auto fromDLPack = at::fromDLPackVersioned;
};

} // namespace at
