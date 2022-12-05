#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ATen_fwd.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <algorithm>

namespace torch {
namespace nested {

/// Nested tensor
///
/// See
/// https://pytorch.org/docs/master/nested.html#torch.nested.nested_tensor
///
/// ```
// implemented on python object to allow torch.nested.nested_tensor to be
// constructed with arbitrarily nested python objects - for now, only arbitrary
// python lists and lists of Tensors
// See torch/csrc/autograd/python_nested_functions_manual.cpp for Python
// implementation
// See here for C++ implementation
inline at::Tensor nested_tensor(
    at::TensorList nested_tensor_data,
    const at::TensorOptions& options = {}) {
  auto out = at::_nested_tensor_from_tensor_list(
      nested_tensor_data,
      c10::typeMetaToScalarType(options.dtype()),
      c10::nullopt,
      options.device(),
      options.pinned_memory());
  if (options.has_requires_grad() && options.requires_grad()) {
    out.requires_grad_(true);
  }
  return out;
}

inline at::Tensor nested_tensor(
    at::ArrayRef<detail::TensorDataContainer> nested_tensor_data,
    const at::TensorOptions& options = {}) {
  for (const auto& tdc : nested_tensor_data) {
    TORCH_CHECK(
        tdc.is_init_list(),
        "nested_tensor() not implemented for these parameters");
  }
  // Construct a TensorList using nested_tensor_data
  std::vector<at::Tensor> tensor_list(nested_tensor_data.size());
  std::transform(
      nested_tensor_data.begin(),
      nested_tensor_data.end(),
      tensor_list.begin(),
      [&](const detail::TensorDataContainer& tdc) {
        return tdc.convert_to_tensor(options);
      });
  auto out = at::_nested_tensor_from_tensor_list(
      tensor_list,
      c10::typeMetaToScalarType(options.dtype()),
      c10::nullopt,
      options.device(),
      options.pinned_memory());
  if (options.has_requires_grad() && options.requires_grad()) {
    out.requires_grad_(true);
  }
  return out;
}

/// As Nested Tensor
///
/// See
/// https://pytorch.org/docs/master/nested.html#torch.nested.as_nested_tensor
///
/// ```
inline at::Tensor as_nested_tensor(
    at::TensorList list,
    c10::optional<at::ScalarType> dtype = c10::nullopt,
    c10::optional<at::Device> device = c10::nullopt) {
  return at::_nested_tensor_from_tensor_list(
      list, dtype, c10::nullopt, device, c10::nullopt);
}

/// Nested to padded tensor
///
/// See
/// https://pytorch.org/docs/master/nested.html#torch.nested.to_padded_tensor
///
/// ```
inline at::Tensor to_padded_tensor(
    const at::Tensor& self,
    double padding,
    at::OptionalIntArrayRef output_size = c10::nullopt) {
  return at::nested_to_padded_tensor(self, padding, output_size);
}

} // namespace nested
} // namespace torch
