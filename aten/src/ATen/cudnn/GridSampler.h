#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

Tensor cudnn_grid_sampler_forward(
    const Tensor& input, const Tensor& grid);


std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input, const Tensor& grid, const Tensor& grad_output);

}}
