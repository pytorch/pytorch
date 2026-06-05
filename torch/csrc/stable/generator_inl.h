#pragma once

// This file implements generator.h. We separated out the Generator struct so
// that stableivalue_conversions.h can depend on the Generator struct (to define
// the From/To conversions) while the Generator method implementations can in
// turn depend on those conversions, without circular dependencies.

#include <torch/csrc/stable/generator_struct.h>
#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

inline Device Generator::device() const {
  int32_t device_type;
  int32_t device_index;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_generator_get_device(gen_.get(), &device_type, &device_index));
  DeviceType extension_device_type = torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
  return Device(extension_device_type, static_cast<DeviceIndex>(device_index));
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_13_0

HIDDEN_NAMESPACE_END(torch, stable)
