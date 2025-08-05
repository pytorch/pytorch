#pragma once
#include <ATen/core/Tensor.h>

#include <ATen/cuda/CUDAConfig.h> // for the definition of AT_CUDNN_ENABLED

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/cudnn-wrapper.h>

namespace at {
namespace native {

#if AT_CUDNN_ENABLED()

void raw_cudnn_layernorm_forward_out(
    const Tensor& X,
    const Tensor& scale,
    const Tensor& bias,
    float epsilon,
    Tensor* mean,
    Tensor* rstd,
    Tensor* Y,
    int64_t M,
    int64_t N);
void raw_cudnn_layernorm_backward_out(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& mean,
    const Tensor& rstd,
    const Tensor& gamma,
    int64_t M,
    int64_t N,
    Tensor* dX,
    Tensor* dgamma,
    Tensor* dbeta);

#endif
} // namespace native
} // namespace at
