#pragma once
#include <ATen/core/Tensor.h>

namespace at {
namespace native {

// Forward declarations for cuDNN v8 batch normalization functions

std::tuple<Tensor, Tensor, Tensor, Tensor> cudnn_batch_norm(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    bool training,
    double exponential_average_factor,
    double epsilon);

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> cudnn_batch_norm_out(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& running_mean,
    const std::optional<Tensor>& running_var,
    bool training,
    double exponential_average_factor,
    double epsilon,
    Tensor& out,
    Tensor& save_mean,
    Tensor& save_var,
    Tensor& reserve);

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input,
    const Tensor& grad_output,
    const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt,
    const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt,
    const std::optional<Tensor>& save_var_opt,
    double epsilon,
    const Tensor& reserveSpace);

size_t _get_cudnn_batch_norm_reserve_space_size(
    const Tensor& input_t,
    bool training);

} // namespace native
} // namespace at
