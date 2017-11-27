#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

Tensor cudnn_batch_norm_forward(
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, const Tensor& running_mean, const Tensor& running_var,
    const Tensor& save_mean, const Tensor& save_var, bool training,
    double exponential_average_factor, double epsilon);

std::tuple<Tensor, Tensor, Tensor> cudnn_batch_norm_backward(
    const Tensor& input, const Tensor& grad_output, const Tensor& weight,
    const Tensor& save_mean, const Tensor& save_var, bool training,
    double epsilon);

}}
