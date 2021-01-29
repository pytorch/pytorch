#pragma once

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cuda/ATenCUDAGeneral.h>

namespace at { namespace native {

TORCH_CUDA_CU_API cudnnHandle_t getCudnnHandle();
}} // namespace at::native
