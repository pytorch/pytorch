#pragma once
#include <ATen/core/Reduction.h>
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
        "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
        input.sizes());

    if (ndims <= 1) {
      nframe = 1;
      dim = ndims == 0 ? 1 : input.size(0);
      TORCH_CHECK(
          target.dim() <= 1 && target.numel() == dim,
          "inconsistent target size: ", target.sizes(), " for input of size: ",
          input.sizes());
    } else {
      nframe = input.size(0);
      dim = input.size(1);
      TORCH_CHECK(
          target.dim() == 2 && target.size(0) == nframe &&
          target.size(1) == dim,
          "inconsistent target size: ", target.sizes(), " for input of size: ",
          input.sizes());
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

inline void multi_margin_loss_backward_grad_output_shape_check(
    const Tensor& grad_output,
    const Tensor& target,
    int64_t nframe,
    int64_t reduction) {
  if (reduction != at::Reduction::None) {
    return;
  }
  // The backward kernels index grad_output per sample, so it must match
  // the forward output shape: [nframe] for a 1-D target, [] for a 0-D one.
  if (target.dim() > 0) {
    TORCH_CHECK(
        grad_output.dim() == 1 && grad_output.size(0) == nframe,
        "multi_margin_loss_backward: expected grad_output to have shape [",
        nframe, "] when reduction='none', but got ", grad_output.sizes());
  } else {
    TORCH_CHECK(
        grad_output.dim() == 0,
        "multi_margin_loss_backward: expected grad_output to be a 0-D scalar "
        "when reduction='none' with a 0-D target, but got ",
        grad_output.sizes());
  }
}

} // namespace at::native
