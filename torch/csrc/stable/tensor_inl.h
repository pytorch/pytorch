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

HIDDEN_NAMESPACE_END(torch, stable)
