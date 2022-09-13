#pragma once

#include <ATen/ATen.h>
#include <torch/torch.h>

namespace torch {
namespace nested {

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
