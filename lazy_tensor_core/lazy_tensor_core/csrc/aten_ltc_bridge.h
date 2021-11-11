#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

#include "lazy_tensor_core/csrc/tensor.h"

namespace torch_lazy_tensors {
namespace bridge {

c10::optional<LazyTensor> TryGetLtcTensor(const at::Tensor& tensor);

bool IsLtcTensor(const at::Tensor& tensor);

// Extracts the LazyTensor out of our version of at::Tensor. Throws an exception
// if tensor is not a lazy tensor.
LazyTensor GetLtcTensor(const at::Tensor& tensor);

// Replaces the lazy tensor embedded within the TensorImpl with the new version.
void ReplaceLtcTensor(const at::Tensor& tensor, LazyTensor new_ltc_tensor);

// Same as above, applied to a list of tensors.
std::vector<LazyTensor> GetLtcTensors(c10::ArrayRef<at::Tensor> tensors);

// If tensor is a lazy tensor type, returns the LazyTensor embedded within it,
// otherwise creates a new lazy tensor type with tensor as data.
LazyTensor GetOrCreateLtcTensor(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

LazyTensor GetOrCreateLtcTensor(const c10::optional<at::Tensor>& tensor,
                                const torch::lazy::BackendDevice& device);

LazyTensor GetLtcTensorOrCreateForWrappedNumber(const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

// Creates a vector of at::Tensor objects extracted from a list of lazy tensors.
std::vector<at::Tensor> LtcCreateTensorList(const at::TensorList& tensors);

// Creates a vector of c10::optional<at::Tensor> objects extracted from a list
// of optional lazy tensors.
std::vector<c10::optional<at::Tensor>> LtcCreateOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);

void LtcUpdateTensors(c10::ArrayRef<at::Tensor> dest_ltc_tensors,
                      c10::ArrayRef<at::Tensor> source_cpu_tensors,
                      c10::ArrayRef<size_t> indices);

void LtcUpdateTensorsMeta(c10::ArrayRef<at::Tensor> dest_ltc_tensors,
                          c10::ArrayRef<at::Tensor> source_cpu_tensors,
                          c10::ArrayRef<size_t> indices);

// Tries to extract the device out of the lazy tensor. Returns nullopt if the
// input is not a lazy tensor.
c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const at::Tensor& tensor);

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const c10::optional<at::Tensor>& tensor);

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const at::TensorList& tensors);

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const at::TensorOptions& tensor_options);

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(const c10::Device& device);

c10::optional<torch::lazy::BackendDevice> GetLtcDevice(
    const c10::optional<c10::Device>& device = c10::nullopt);

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

std::string ToLtcString(const c10::Device& device);

at::Tensor LtcToAtenTensor(LazyTensor ltc_tensor,
                           const at::TensorOptions& tensor_options);

// Creates an ATen tensor from an LazyTensor.
at::Tensor AtenFromLtcTensor(LazyTensor ltc_tensor);

std::vector<at::Tensor> AtenFromLtcTensors(
    c10::ArrayRef<LazyTensor> ltc_tensors);

// Creates a lazy tensor holding the data in tensor, on the given device.
at::Tensor CreateLtcTensor(at::Tensor tensor,
                           const c10::optional<torch::lazy::BackendDevice>& device);

// Given a vector of at::Tensor creates a vector of lazy tensors on the given
// device.
std::vector<at::Tensor> CreateLtcTensors(const std::vector<at::Tensor>& tensors,
                                         const c10::optional<torch::lazy::BackendDevice>& device);

// Returns true if the tensor is a view created via interoperability.
bool IsInteropView(const at::Tensor& t);

template <size_t... Indices>
auto TupleAtenFromLtcTensorsImpl(const std::vector<LazyTensor>& tensors, std::index_sequence<Indices...>) {
    return std::make_tuple(AtenFromLtcTensor(tensors[Indices])...);
}

template <size_t N>
auto TupleAtenFromLtcTensors(const std::vector<LazyTensor>& tensors) {
    return TupleAtenFromLtcTensorsImpl(tensors, std::make_index_sequence<N>{});
}

}  // namespace bridge
}  // namespace torch_lazy_tensors
