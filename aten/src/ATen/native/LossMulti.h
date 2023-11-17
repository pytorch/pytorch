#pragma once
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

namespace at::native {
namespace {
  static C10_UNUSED void multilabel_margin_loss_shape_check(
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

  static C10_UNUSED void multi_margin_loss_shape_check(
    int64_t& nframe,
    int64_t& dim,
    const int64_t& ndims,
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight) {
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


}  // anonymous namespace
} // namespace at::native
