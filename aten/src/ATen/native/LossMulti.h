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
        "Expected input to be a scalar, non-empty 1D tensor, or non-empty 2D "
        "tensor, but got input with ",
        ndims,
        " dimensions and shape ",
        input.sizes());

    if (ndims <= 1) {
      nframe = 1;
      dim = ndims == 0 ? 1 : input.size(0);
      TORCH_CHECK(
          target.dim() <= 1 && target.numel() == dim,
          "Expected target to have at most 1 dimension and ",
          dim,
          " elements to match input, but got target with shape ",
          target.sizes());
    } else {
      nframe = input.size(0);
      dim = input.size(1);
      TORCH_CHECK(
          target.dim() == 2 && target.size(0) == nframe &&
          target.size(1) == dim,
          "Expected target to be 2D with shape [",
          nframe,
          ", ",
          dim,
          "] to match input, but got target with ",
          target.dim(),
          " dimensions and shape ",
          target.sizes());
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
        "multi_margin_loss: target tensor should be 1-D with size equal to "
        "the number of input samples (batch size). Expected target size [",
        nframe, "], but got ", target.sizes(),
        ". Input has shape ", input.sizes(), ".");
    if (weight && weight->defined()) {
      TORCH_CHECK(
          weight->dim() <= 1 && weight->numel() == dim,
          "inconsistent weight size, expected ", dim, " but got ",
          weight->sizes());
    }
}

} // namespace at::native
