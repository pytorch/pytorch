#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/macros/Export.h>
#include <torch/csrc/lazy/backend/backend_device.h>

namespace torch {
namespace lazy {

TORCH_API bool in_lazy_mode();
TORCH_API c10::DispatchKey device_to_dispatch_key(c10::DeviceType device_type);
TORCH_API void LazyModeEnter(c10::Device device);
TORCH_API void LazyModeExit(c10::Device device);
TORCH_API c10::DispatchKey GetUnlazyDispatchKey();

at::Tensor PrepareTensorForMetaKernel(at::Tensor tensor, BackendDevice lazy_device);

} // namespace lazy
} // namespace torch
