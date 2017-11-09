#pragma once

#include "THC/THC.h"

#include <ATen/ATen.h>

#include "cudnn-wrapper.h"

namespace torch { namespace cudnn {

void cudnn_grid_sampler_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& grid, const at::Tensor& output);


void cudnn_grid_sampler_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& grad_input,
    const at::Tensor& grid, const at::Tensor& grad_grid,
    const at::Tensor& grad_output);

}}
