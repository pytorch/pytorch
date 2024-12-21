#pragma once
#include <ATen/core/Tensor.h>

namespace at::native::mps {

void _fused_adamw_amsgrad_mps_impl_(
    TensorList params,
    TensorList grads,
    TensorList exp_avgs,
    TensorList exp_avg_sqs,
    TensorList max_exp_avg_sqs,
    TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<Tensor>& grad_scale,
    const std::optional<Tensor>& found_inf);

void _fused_adamw_amsgrad_mps_impl_(
    TensorList params,
    TensorList grads,
    TensorList exp_avgs,
    TensorList exp_avg_sqs,
    TensorList max_exp_avg_sqs,
    TensorList state_steps,
    const Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const std::optional<Tensor>& grad_scale,
    const std::optional<Tensor>& found_inf);
} // namespace at::native::mps
