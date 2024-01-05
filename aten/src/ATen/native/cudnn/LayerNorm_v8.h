#pragma once
#include <ATen/core/Tensor.h>

#include <ATen/cuda/CUDAConfig.h>  // for the definition of AT_CUDNN_ENABLED

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>

namespace at { namespace native {


#if AT_CUDNN_ENABLED()

void raw_cudnn_layernorm_forward_out(const Tensor& X, const Tensor& scale, const Tensor& bias, float epsilon,  Tensor* mean,  Tensor* rstd);

#endif
}}
