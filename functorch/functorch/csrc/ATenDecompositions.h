#pragma once

#include <ATen/ATen.h>

namespace at { namespace functorch {

// TODO: Figure out how to delete all of these and replace with
// with the "official" decompositions that are written in Python.

inline Tensor nll_loss_backward_decomp(
    const at::Tensor & grad_output, const at::Tensor & self,
    const at::Tensor & target, const c10::optional<at::Tensor> & weight,
    int64_t reduction, int64_t ignore_index, const at::Tensor & total_weight) {

  int64_t channel_dim = 1;
  if (self.dim() < 2) {
    channel_dim = 0;
  }
  auto self_ = self;
  auto target_ = target.unsqueeze(channel_dim);

  auto grad_output_ = grad_output;
  if (reduction == Reduction::Mean) {
    grad_output_ = grad_output_ / total_weight;
  }

  auto grad_input = at::zeros_like(self);
  grad_input = at::scatter(grad_input, channel_dim, target_, -1.0);

  if (grad_output_.dim() < grad_input.dim() && grad_output_.dim() > 0) {
    grad_output_ = grad_output_.unsqueeze(channel_dim);
  }

  Tensor weight_;
  if (weight && weight->defined()) {
    auto shape = weight->sizes();
    std::vector<int64_t> new_shape(self_.dim(), 1);
    new_shape[channel_dim] = shape[0];
    weight_ = weight->reshape(new_shape);
    grad_output_ = grad_output_ * weight_;
  }

  bool has_ignore_index = ignore_index >= 0;
  Tensor ignore_index_mask;
  if (has_ignore_index) {
    ignore_index_mask = target_ != ignore_index;
    grad_output_ = grad_output_ * ignore_index_mask;
  }

  return grad_input * grad_output_;
}

}} // namespace at::functorch
