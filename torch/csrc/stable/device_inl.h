#pragma once

// This file implements device.h. We separated out the Device struct so that
// other files can depend on the Device struct (like stableivalue_conversions.h)
// and the implementations of the Device methods can depend on APIs in
// stableivalue_conversions.h without circular dependencies.

#include <torch/csrc/stable/device_struct.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/macros/Macros.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

inline DeviceType Device::type() const {
  int32_t device_type;
  TORCH_ERROR_CODE_CHECK(torch_device_type(dh_.get(), &device_type));
  return torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
}

HIDDEN_NAMESPACE_END(torch, stable)
