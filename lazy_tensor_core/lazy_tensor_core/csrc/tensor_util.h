#pragma once

#include <string>
#include <vector>

#include "lazy_tensor_core/csrc/compiler/backend_impl_interface.h"
#include "lazy_tensor_core/csrc/device.h"
#include "lazy_tensors/literal.h"
#include "lazy_tensors/shape.h"
#include "lazy_tensors/span.h"
#include "torch/csrc/autograd/variable.h"

namespace torch_lazy_tensors {

std::vector<int64_t> ComputeArrayStrides(c10::ArrayRef<int64_t> sizes);

std::vector<at::Tensor> DataHandlesToTensors(
    c10::ArrayRef<compiler::DataPtr> data_handles,
    at::ScalarType dest_element_type);

bool TensorCompare(const at::Tensor& t1, const at::Tensor& t2);

// Uploads an ATEN tensor data to the device and fetches the corresponding
// device data handle.
compiler::DataPtr TensorToDataHandle(
    const at::Tensor& tensor, const Device& device);

torch::lazy::hash_t TensorHash(const at::Tensor& tensor);

// Retrieves the device data handles by parallel uploading data onto the
// corresponding devices.
std::vector<compiler::DataPtr> CreateTensorsData(
    const std::vector<at::Tensor>& tensors,
    const std::vector<std::string>& devices);

template<typename... TupleType>
std::vector<std::vector<int64_t>> CreateComputationShapeFromMetaTensors(const std::tuple<TupleType...>& tensors) {
  std::vector<std::vector<int64_t>> shape;
  c10::guts::apply([&shape] (const auto&... tensors) {
      ((shape.push_back(tensors.sizes().vec())), ...);
  }, tensors);
  return shape;
}

template<typename... TupleType>
std::vector<at::ScalarType> CreateDTypeFromMetaTensors(const std::tuple<TupleType...>& tensors) {
  std::vector<at::ScalarType> dtypes;
  c10::guts::apply([&dtypes] (const auto&... tensors) {
      ((dtypes.push_back(tensors.scalar_type())), ...);
  }, tensors);
  return dtypes;
}

// Routing values to device data maximizes the changes for compilation cache
// hits, but it can prevent the compiler to perform optimizations. So tensor
// values which are within a given set, are routed to constant scalars if this
// API returns true.
bool IsSpecialScalar(const at::Scalar& value);

}  // namespace torch_lazy_tensors
