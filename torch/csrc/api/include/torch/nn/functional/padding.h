#pragma once

#include <torch/nn/options/padding.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor _narrow_with_range(const Tensor& input, int64_t dim, int64_t start, int64_t end) {
  return input.narrow(dim, start, end - start);
}

inline Tensor _pad_circular(Tensor input, IntArrayRef padding) {
  input = torch::cat({input, _narrow_with_range(input, 2, 0, padding[-1 + padding.size()])}, /*dim=*/2);
  input = torch::cat({_narrow_with_range(input, 2, -(padding[-1 + padding.size()] + padding[-2 + padding.size()]), -padding[-1 + padding.size()]), input}, /*dim=*/2);

  if (padding.size() > 2) {
    input = torch::cat({input, _narrow_with_range(input, 3, 0, padding[-3 + padding.size()])}, /*dim=*/3);
    input = torch::cat({_narrow_with_range(input, 3, -(padding[-3 + padding.size()] + padding[-4 + padding.size()]), -padding[-3 + padding.size()]), input}, /*dim=*/3);
  }

  if (padding.size() > 4) {
    input = torch::cat({input, _narrow_with_range(input, 4, 0, padding[-5 + padding.size()])}, /*dim=*/4);
    input = torch::cat({_narrow_with_range(input, 4, -(padding[-5 + padding.size()] + padding[-6 + padding.size()]), -padding[-5 + padding.size()]), input}, /*dim=*/4);
  }

  return input;
}

inline Tensor pad(const Tensor& input, const PadFuncOptions& options) {
  TORCH_CHECK(options.pad().size() % 2 == 0, "Padding length must be divisible by 2");
  TORCH_CHECK(((int64_t)(options.pad().size() / 2)) <= input.dim(), "Padding length too large");
  if (c10::get_if<enumtype::kConstant>(&options.mode())) {
    return torch::constant_pad_nd(input, options.pad(), options.value());
  } else {
    TORCH_CHECK(
      options.value() == 0,
      "Padding mode \"",
      torch::enumtype::get_enum_name(options.mode()),
      "\" doesn't take in value argument");
    if (input.dim() == 3) {
      TORCH_CHECK(options.pad().size() == 2, "3D tensors expect 2 values for padding");
      if (c10::get_if<enumtype::kReflect>(&options.mode())) {
        return torch::reflection_pad1d(input, options.pad());
      } else if (c10::get_if<enumtype::kReplicate>(&options.mode())) {
        return torch::replication_pad1d(input, options.pad());
      } else if (c10::get_if<enumtype::kCircular>(&options.mode())) {
        return _pad_circular(input, options.pad());
      } else {
        TORCH_CHECK(false, "NotImplementedError");
      }
    } else if (input.dim() == 4) {
      TORCH_CHECK(options.pad().size() == 4, "4D tensors expect 4 values for padding");
      if (c10::get_if<enumtype::kReflect>(&options.mode())) {
        return torch::reflection_pad2d(input, options.pad());
      } else if (c10::get_if<enumtype::kReplicate>(&options.mode())) {
        return torch::replication_pad2d(input, options.pad());
      } else if (c10::get_if<enumtype::kCircular>(&options.mode())) {
        return _pad_circular(input, options.pad());
      } else {
        TORCH_CHECK(false, "NotImplementedError");
      }
    } else if (input.dim() == 5) {
      TORCH_CHECK(options.pad().size() == 6, "5D tensors expect 6 values for padding");
      if (c10::get_if<enumtype::kReflect>(&options.mode())) {
        TORCH_CHECK(false, "NotImplementedError");
      } else if (c10::get_if<enumtype::kReplicate>(&options.mode())) {
        return torch::replication_pad3d(input, options.pad());
      } else if (c10::get_if<enumtype::kCircular>(&options.mode())) {
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
