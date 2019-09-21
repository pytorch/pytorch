#pragma once

#include <torch/nn/options/padding.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor _pad_circular(Tensor input, IntArrayRef padding) {
  input = torch::cat({input, input.index_select(2, torch::arange(0, padding[-1]))}, /*dim=*/2);
  input = torch::cat({input.index_select(2, torch::arange(-(padding[-1] + padding[-2]), -padding[-1])), input}, /*dim=*/2);

  if (padding.size() > 2) {
    input = torch::cat({input, input.index_select(3, torch::arange(0, padding[-3]))}, /*dim=*/3);
    input = torch::cat({input.index_select(3, torch::arange(-(padding[-3] + padding[-4]), -padding[-3])), input}, /*dim=*/3);
  }

  if (padding.size() > 4) {
    input = torch::cat({input, input.index_select(4, torch::arange(0, padding[-5]))}, /*dim=*/4);
    input = torch::cat({input.index_select(4, torch::arange(-(padding[-5] + padding[-6]), -padding[-5])), input}, /*dim=*/4);
  }

  return input;
}

inline Tensor pad(const Tensor& input, const PadOptions& options) {
  TORCH_CHECK(options.pad().size() % 2 == 0, "Padding length must be divisible by 2");
  TORCH_CHECK(options.pad().size() / 2 <= input.dim(), "Padding length too large");
  if (options.mode() == "constant") {
    return torch::constant_pad_nd(input, options.pad(), options.value());
  } else {
    TORCH_CHECK(options.value() == 0, "Padding mode \"", mode, "\" doesn't take in value argument");
    if (input.dim() == 3) {
      TORCH_CHECK(options.pad().size() == 2, "3D tensors expect 2 values for padding");
      if (options.mode() == "reflect") {
        return torch::reflection_pad1d(input, options.pad());
      } else if (options.mode() == "replicate") {
        return torch::replication_pad1d(input, options.pad());
      } else if (options.mode() == "circular") {
        return _pad_circular(input, options.pad());
      } else {
        TORCH_CHECK(false, "NotImplementedError");
      }
    } else if (input.dim() == 4) {
      TORCH_CHECK(options.pad().size() == 4, "4D tensors expect 4 values for padding");
      if (options.mode() == "reflect") {
        return torch::reflection_pad2d(input, options.pad());
      } else if (options.mode() == "replicate") {
        return torch::replication_pad2d(input, options.pad());
      } else if (options.mode() == "circular") {
        return _pad_circular(input, options.pad());
      } else {
        TORCH_CHECK(false, "NotImplementedError");
      }
    } else if (input.dim() == 5) {
      TORCH_CHECK(options.pad().size() == 6, "5D tensors expect 6 values for padding");
      if (options.mode() == "reflect") {
        TORCH_CHECK(false, "NotImplementedError");
      } else if (options.mode() == "replicate") {
        return torch::replication_pad3d(input, options.pad());
      } else if (options.mode() == "circular") {
        return _pad_circular(input, options.pad());
      } else {
        TORCH_CHECK(false, "NotImplementedError");
      }
    } else {
      TORCH_CHECK(false, "Only 3D, 4D, 5D padding with non-constant padding are supported for now");
    }
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
