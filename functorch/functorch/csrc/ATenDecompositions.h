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

Tensor _log_softmax_backward_data_decomp(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  auto result = grad_output - at::exp(output) * at::sum(grad_output, dim, /*keepdim=*/true);
  return result.to(input_dtype);
}

Tensor _softmax_backward_data_decomp(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  auto new_grad = grad_output * output;
  auto result = (new_grad - output * at::sum(new_grad, dim, /*keepdim*/true));
  return result.to(input_dtype);
}

Tensor mse_loss_backward_decomp(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction) {
  auto norm = reduction == Reduction::Mean ? 2. / input.numel() : 2.;
  return norm * (input - target) * grad_output;
}

Tensor l1_loss_backward_decomp(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto sign = at::sign(self - target);
  auto norm = reduction == Reduction::Mean ? sign / self.numel() : sign;
  return grad_output * norm;
}

Tensor binary_cross_entropy_backward_decomp(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const optional<Tensor>& weight,
    int64_t reduction) {
  auto EPSILON = 1e-12;
  auto result = grad_output * (self - target) / at::clamp_min(self * (1 - self), EPSILON);
  if (weight.has_value() && weight->defined()) {
    result = result * weight.value();
  }
  if (reduction == Reduction::Mean) {
    result = result / self.numel();
  }
  return result;
}

}} // namespace at::functorch
