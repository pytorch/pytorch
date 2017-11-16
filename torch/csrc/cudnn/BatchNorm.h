#pragma once

#include "cudnn-wrapper.h"
#include <ATen/ATen.h>
#include "THC/THC.h"


namespace torch { namespace cudnn {

void cudnn_batch_norm_forward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& output, const at::Tensor& weight,
    const at::Tensor& bias, const at::Tensor& running_mean, const at::Tensor& running_var,
    const at::Tensor& save_mean, const at::Tensor& save_var, bool training,
    double exponential_average_factor, double epsilon);

void cudnn_batch_norm_backward(
    THCState* state, cudnnHandle_t handle, cudnnDataType_t dataType,
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& grad_input,
    const at::Tensor& grad_weight, const at::Tensor& grad_bias, const at::Tensor& weight,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    const at::Tensor& save_mean, const at::Tensor& save_var, bool training,
    double epsilon);

}}
