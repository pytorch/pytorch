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
// at::Tensor => LazyTensor
//////////////////////////////////////////////////////////////////////////////

bool IsLtcTensor(const at::Tensor& tensor);
c10::optional<LazyTensor> TryGetLtcTensor(const at::Tensor& tensor);

// Extracts the LazyTensor out of our version of at::Tensor. Throws an exception
// if tensor is not a lazy tensor.
LazyTensor GetLtcTensor(const at::Tensor& tensor);

// Same as above, applied to a list of tensors.
std::vector<LazyTensor> GetLtcTensors(c10::ArrayRef<at::Tensor> tensors);

// If tensor is a lazy tensor type, returns the LazyTensor embedded within it,
// otherwise creates a new lazy tensor type with tensor as data.
LazyTensor GetOrCreateLtcTensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

LazyTensor GetOrCreateLtcTensor(const c10::optional<at::Tensor>& tensor,
                                const torch::lazy::BackendDevice& device);

LazyTensor GetLtcTensorOrCreateForWrappedNumber(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

// Creates a lazy tensor holding the data in tensor, on the given device.
at::Tensor CreateLtcTensor(at::Tensor tensor,
                           const c10::optional<torch::lazy::BackendDevice>& device);

// Creates a vector of at::Tensor objects extracted from a list of lazy tensors.
std::vector<at::Tensor> LtcCreateTensorList(const at::TensorList& tensors);

//////////////////////////////////////////////////////////////////////////////
// LazyTensor => at::Tensor
//////////////////////////////////////////////////////////////////////////////

// Creates an ATen tensor from an LazyTensor.
at::Tensor AtenFromLtcTensor(LazyTensor ltc_tensor);

template <size_t... Indices>
auto TupleAtenFromLtcTensorsImpl(const std::vector<LazyTensor>& tensors, std::index_sequence<Indices...>) {
    return std::make_tuple(AtenFromLtcTensor(tensors[Indices])...);
}

template <size_t N>
auto TupleAtenFromLtcTensors(const std::vector<LazyTensor>& tensors) {
    return TupleAtenFromLtcTensorsImpl(tensors, std::make_index_sequence<N>{});
}

//////////////////////////////////////////////////////////////////////////////
// Device Management
//////////////////////////////////////////////////////////////////////////////

// Tries to extract the device out of the lazy tensor. Returns nullopt if the
// input is not a lazy tensor.
c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const at::Tensor& tensor);

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const c10::optional<c10::Device>& device = c10::nullopt);

template<typename T, typename... Args>
c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const T& tensor, const Args&... forward_tensors) {
    auto optional_device = GetLtcDevice(tensor);
    if (optional_device) {
        return optional_device;
    }
    return GetLtcDevice(forward_tensors...);
}

torch::lazy::BackendDevice AtenDeviceToLtcDevice(const c10::Device& device);

c10::Device LtcDeviceToAtenDevice(const torch::lazy::BackendDevice& device);

}  // namespace bridge
}  // namespace torch_lazy_tensors
