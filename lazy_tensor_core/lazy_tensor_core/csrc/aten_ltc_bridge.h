#pragma once

#include <ATen/Device.h>
#include <ATen/Functions.h>
#include <ATen/Tensor.h>

#include <vector>

#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensor_core/csrc/tensor.h"
#include "lazy_tensors/span.h"

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
std::vector<LazyTensor> GetLtcTensors(
    lazy_tensors::Span<const at::Tensor> tensors);

// If tensor is a lazy tensor type, returns the LazyTensor embedded within it,
// otherwise creates a new lazy tensor type with tensor as data.
LazyTensor GetOrCreateLtcTensor(const at::Tensor& tensor, const Device& device);

LazyTensor GetOrCreateLtcTensor(const c10::optional<at::Tensor>& tensor,
                                const Device& device);

// Creates a vector of at::Tensor objects extracted from a list of lazy tensors.
std::vector<at::Tensor> LtcCreateTensorList(const at::TensorList& tensors);

// Creates a vector of c10::optional<at::Tensor> objects extracted from a list
// of optional lazy tensors.
std::vector<c10::optional<at::Tensor>> LtcCreateOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors);

void LtcUpdateTensors(lazy_tensors::Span<const at::Tensor> dest_ltc_tensors,
                      lazy_tensors::Span<const at::Tensor> source_cpu_tensors,
                      lazy_tensors::Span<const size_t> indices);

void LtcUpdateTensorsMeta(
    lazy_tensors::Span<const at::Tensor> dest_ltc_tensors,
    lazy_tensors::Span<const at::Tensor> source_cpu_tensors,
    lazy_tensors::Span<const size_t> indices);

// Tries to extract the device out of the lazy tensor. Returns nullopt if the
// input is not a lazy tensor.
c10::optional<Device> GetLtcDevice(const at::Tensor& tensor);

c10::optional<Device> GetLtcDevice(const c10::optional<at::Tensor>& tensor);

c10::optional<Device> GetLtcDevice(const at::TensorList& tensors);

c10::optional<Device> GetLtcDevice(const at::TensorOptions& tensor_options);

c10::optional<Device> GetLtcDevice(const c10::Device& device);

c10::optional<Device> GetLtcDevice(
    const c10::optional<c10::Device>& device = c10::nullopt);

Device AtenDeviceToLtcDevice(const c10::Device& device);

c10::Device LtcDeviceToAtenDevice(const Device& device);

std::string ToLtcString(const c10::Device& device);

c10::Device AtenDefaultDevice();

c10::Device SetCurrentDevice(const c10::Device& device);

Device SetCurrentDevice(const Device& device);

c10::Device GetCurrentAtenDevice();

at::Tensor LtcToAtenTensor(LazyTensor ltc_tensor,
                           const at::TensorOptions& tensor_options);

// Creates an ATen tensor from an LazyTensor.
at::Tensor AtenFromLtcTensor(LazyTensor ltc_tensor);

std::vector<at::Tensor> AtenFromLtcTensors(
    lazy_tensors::Span<const LazyTensor> ltc_tensors);

// Creates a lazy tensor holding the data in tensor, on the given device.
at::Tensor CreateLtcTensor(at::Tensor tensor,
                           const c10::optional<Device>& device);

// Given a vector of at::Tensor creates a vector of lazy tensors on the given
// device.
std::vector<at::Tensor> CreateLtcTensors(const std::vector<at::Tensor>& tensors,
                                         const c10::optional<Device>& device);

// Returns true if the tensor is a view created via interoperability.
bool IsInteropView(const at::Tensor& t);

}  // namespace bridge
}  // namespace torch_lazy_tensors
