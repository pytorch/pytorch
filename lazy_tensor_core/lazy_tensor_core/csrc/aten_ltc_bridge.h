#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/csrc/lazy/backend/backend_device.h>

namespace torch_lazy_tensors {
namespace bridge {
//////////////////////////////////////////////////////////////////////////////
// Device Management
//////////////////////////////////////////////////////////////////////////////

// Tries to extract the backend device out of the lazy tensor. Returns nullopt if the
// input is not a lazy tensor.
c10::optional<torch::lazy::BackendDevice> GetBackendDevice(const at::Tensor& tensor);

// For variadic template.
c10::optional<torch::lazy::BackendDevice> GetBackendDevice();

template<typename T, typename... Args>
c10::optional<torch::lazy::BackendDevice> GetBackendDevice(const T& tensor, const Args&... forward_tensors) {
    auto optional_device = GetBackendDevice(tensor);
    if (optional_device) {
        return optional_device;
    }
    return GetBackendDevice(forward_tensors...);
}

torch::lazy::BackendDevice AtenDeviceToBackendDevice(const c10::Device& device);

c10::Device BackendDeviceToAtenDevice(const torch::lazy::BackendDevice& device);

}  // namespace bridge
}  // namespace torch_lazy_tensors
