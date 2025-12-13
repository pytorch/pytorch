#pragma once

// This file implements tensor.h. We separated out the Tensor struct so that
// other files can depend on the Tensor struct (like library.h) and the
// implementations of the Tensor methods can depend on APIs in library.h
// without circular dependencies.

#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

using torch::headeronly::ScalarType;

inline ScalarType Tensor::scalar_type() const {
  int32_t dtype;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(ath_.get(), &dtype));
  return torch::stable::detail::to<ScalarType>(
      torch::stable::detail::from(dtype));
}

inline Device Tensor::device() const {
  int32_t device_type;
  int32_t device_index;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(ath_.get(), &device_type));
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(ath_.get(), &device_index));
  DeviceType extension_device_type = torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
  return Device(extension_device_type, static_cast<DeviceIndex>(device_index));
}

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
// The following data ptr cast methods mirror the methods defined in
// aten/src/ATen/templates/TensorMethods.cpp
#define DEFINE_DATA_PTR_CAST(T, name, PRED)               \
  template <>                                             \
  inline T* Tensor::mutable_data_ptr() const {            \
    auto stype = scalar_type();                           \
    STD_TORCH_CHECK(                                      \
        PRED(stype, torch::headeronly::ScalarType::name), \
        "expected scalar type " #name " but found ",      \
        torch::headeronly::toString(stype));              \
    return static_cast<T*>(mutable_data_ptr());           \
  }                                                       \
  template <>                                             \
  inline const T* Tensor::const_data_ptr() const {        \
    auto stype = scalar_type();                           \
    STD_TORCH_CHECK(                                      \
        PRED(stype, torch::headeronly::ScalarType::name), \
        "expected scalar type " #name " but found ",      \
        torch::headeronly::toString(stype));              \
    return static_cast<const T*>(const_data_ptr());       \
  }

#define _PRED(S1, S2) S1 == S2
#define DEFINE_CAST(T, name) DEFINE_DATA_PTR_CAST(T, name, _PRED)
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CAST)
DEFINE_CAST(uint16_t, UInt16)
DEFINE_CAST(uint32_t, UInt32)
DEFINE_CAST(uint64_t, UInt64)
#undef DEFINE_CAST
#undef _PRED
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

HIDDEN_NAMESPACE_END(torch, stable)
