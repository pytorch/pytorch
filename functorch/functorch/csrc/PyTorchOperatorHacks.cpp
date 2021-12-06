#include <functorch/csrc/DynamicLayer.h>
#include <functorch/csrc/Constants.h>
#include <torch/library.h>
#include <ATen/ATen.h>

namespace at { namespace functorch {

// TODO: all of these should be fixed in a more blessed way. In particular,
// it is bad if any of these go out-of-sync with the implementations in
// pytorch/pytorch.
//
// This file contains hacks for composite PyTorch operators that are problematic.
// For example, the composite op might have in-place operations,
// or call data_ptr. We have some idea of how to fix these things in the long term
// (e.g. functionalization for the in-place operations).

// TODO: can replace with better conditional functionalization
static Tensor value_selecting_reduction_backward_hack(
    const Tensor& grad,
    int64_t dim,
    const Tensor& indices,
    IntArrayRef sizes,
    bool keepdim) {
  if (!keepdim && sizes.size() > 0) {
    auto grad_ = grad.unsqueeze(dim);
    auto indices_ = indices.unsqueeze(dim);
    return at::zeros(sizes, grad_.options()).scatter(dim, indices_, grad_);
  }
  return at::zeros(sizes, grad.options()).scatter(dim, indices, grad);
}

TORCH_LIBRARY_IMPL(aten, FT_DYNAMIC_LAYER_FRONT_MODE_KEY, m) {
  m.impl("value_selecting_reduction_backward", value_selecting_reduction_backward_hack);
}

}}
