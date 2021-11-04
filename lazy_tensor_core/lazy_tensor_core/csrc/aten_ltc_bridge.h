#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

#include "c10/util/Optional.h"
#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace bridge {
//////////////////////////////////////////////////////////////////////////////
// Device Management
//////////////////////////////////////////////////////////////////////////////

// Tries to extract the device out of the lazy tensor. Returns nullopt if the
// input is not a lazy tensor.
c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const at::Tensor& tensor);

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const c10::optional<c10::Device>& device = c10::nullopt);

c10::optional<torch::lazy::BackendDevice> GetSameBackendDeviceOrUseDefault();
c10::optional<torch::lazy::BackendDevice> GetSameBackendDeviceOrUseDefault(const at::Tensor& tensor);
c10::optional<torch::lazy::BackendDevice> GetSameBackendDeviceOrUseDefault(const at::TensorList& tensors);

template<typename T, typename... Args>
c10::optional<torch::lazy::BackendDevice> GetSameBackendDeviceOrUseDefault(const T& tensor, const Args&... forward_tensors) {
    auto optional_device = GetSameBackendDeviceOrUseDefault(tensor);
    if (optional_device) {
        return optional_device;
    }
    return GetSameBackendDeviceOrUseDefault(forward_tensors...);
}

torch::lazy::BackendDevice AtenDeviceToLtcDevice(const c10::Device& device);

c10::Device LtcDeviceToAtenDevice(const torch::lazy::BackendDevice& device);

}  // namespace bridge
}  // namespace torch_lazy_tensors
