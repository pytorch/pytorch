#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace torch {
namespace nested {

/// Nested tensor
///
/// See
/// https://pytorch.org/docs/master/nested.html#torch.nested.nested_tensor
///
/// ```
inline Tensor nested_tensor(
    TensorList list,
    c10::optional<ScalarType> dtype = c10::nullopt,
    c10::optional<Device> device = c10::nullopt,
    c10::optional<bool> requires_grad = false,
    c10::optional<bool> pin_memory = false) {
  std::vector<Tensor> new_list;
  for (const auto i : c10::irange(list.size())) {
    new_list.push_back(list[i].clone().detach());
  }
  auto out = torch::_nested_tensor_from_tensor_list(
      new_list, dtype, c10::nullopt, device, pin_memory);
  if (requires_grad.has_value() && requires_grad.value()) {
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
inline Tensor as_nested_tensor(
    TensorList list,
    c10::optional<ScalarType> dtype = c10::nullopt,
    c10::optional<Device> device = c10::nullopt) {
  return at::_nested_tensor_from_tensor_list(
      list, dtype, c10::nullopt, device, c10::nullopt);
}

/// Nested to padded tensor
///
/// See
/// https://pytorch.org/docs/master/nested.html#torch.nested.to_padded_tensor
///
/// ```
inline Tensor to_padded_tensor(
    const Tensor& self,
    double padding,
    OptionalIntArrayRef output_size = c10::nullopt) {
  return torch::nested_to_padded_tensor(self, padding, output_size);
}

} // namespace nested
} // namespace torch
