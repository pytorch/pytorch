#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

namespace at::native {
  inline void multilabel_margin_loss_shape_check(
    int64_t& nframe,
    int64_t& dim,
    const int64_t& ndims,
    const Tensor& input,
    const Tensor& target) {
    TORCH_CHECK(
        (ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0,
        "Expected input tensor to have 0, 1 or 2 dimension, but got: ", ndims, ", with shape: ",
        input.sizes());

    if (ndims <= 1) {
      nframe = 1;
      dim = ndims == 0 ? 1 : input.size(0);
      TORCH_CHECK(
          target.dim() <= 1 && target.numel() == dim,
          "Expected target tensor to have 0 or 1 dimension, and ", dim, " elements, but got: dim = ", target.dim(),
          ", numel = ", target.numel());
    } else {
      nframe = input.size(0);
      dim = input.size(1);
      TORCH_CHECK(
          target.dim() == 2 && target.size(0) == nframe &&
          target.size(1) == dim,
          "Expected target tensor to have 2 dimension and shape [", nframe, ", ", dim,
          "], but got: dim = ", target.dim(), ", shape = ", target.sizes());
    }
  }

  inline void multi_margin_loss_shape_check(
    int64_t& nframe,
    int64_t& dim,
    const int64_t& ndims,
    const Tensor& input,
    const Tensor& target,
    const std::optional<Tensor>& weight) {
    TORCH_CHECK(
        (ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0,
        "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
        input.sizes());

    if (ndims <= 1) {
      nframe = 1;
      dim = ndims == 0 ? 1 : input.size(0);
    } else {
      nframe = input.size(0);
      dim = input.size(1);
    }

    TORCH_CHECK(
        target.dim() <= 1 && target.numel() == nframe,
        "inconsistent target size, expected ", nframe, " but got ",
        target.sizes());
    if (weight && weight->defined()) {
      TORCH_CHECK(
          weight->dim() <= 1 && weight->numel() == dim,
          "inconsistent weight size, expected ", dim, " but got ",
          weight->sizes());
    }
}

} // namespace at::native
