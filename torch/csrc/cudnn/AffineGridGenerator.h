#pragma once

#include "cudnn-wrapper.h"
#include "THC/THC.h"
#include <ATen/ATen.h>

namespace torch { namespace cudnn {

void cudnn_affine_grid_generator_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& theta, const at::Tensor& grid,
    int N, int C, int H, int W);

void cudnn_affine_grid_generator_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& grad_theta, const at::Tensor& grad_grid,
    int N, int C, int H, int W);

}}
