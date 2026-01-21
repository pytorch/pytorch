#pragma once

// This file implements device.h. We separated out the Device struct so that
// other files can depend on the Device struct (like stableivalue_conversions.h)
// and the implementations of the Device methods can depend on APIs in
// stableivalue_conversions.h without circular dependencies.

#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/device_struct.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/version.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/shim_utils.h>

#include <string>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

using DeviceType = torch::headeronly::DeviceType;
using DeviceIndex = torch::stable::accelerator::DeviceIndex;

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

inline Device::Device(const std::string& device_string) {
  uint32_t device_type;
  int32_t device_index;

  TORCH_ERROR_CODE_CHECK(torch_parse_device_string(
      device_string.c_str(), &device_type, &device_index));

  DeviceType dt = torch::stable::detail::to<DeviceType>(
      torch::stable::detail::from(device_type));
  DeviceIndex di = static_cast<DeviceIndex>(device_index);

  *this = Device(dt, di);
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

HIDDEN_NAMESPACE_END(torch, stable)
