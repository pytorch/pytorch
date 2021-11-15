#pragma once

#include <string>
#include <vector>

#include <torch/csrc/lazy/core/shape.h>

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensors/literal.h"
#include "lazy_tensors/span.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {

std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes);

std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<compiler::BackendDataPtr> data_handles,
    at::ScalarType dest_element_type);

bool TensorCompare(const at::Tensor& t1, const at::Tensor& t2);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
compiler::BackendDataPtr TensorToDataHandle(
    const at::Tensor& tensor, const torch::lazy::BackendDevice& device);

torch::lazy::hash_t TensorHash(const at::Tensor& tensor);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
std::vector<compiler::BackendDataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<torch::lazy::BackendDevice>& devices);

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
bool IsSpecialScalar(const at::Scalar& value);

}  // namespace torch_lazy_tensors
