#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>

#pragma once

namespace at {
namespace native {

Tensor embedding_backward_cuda_kernel(
    const Tensor &grad,
    const Tensor &orig_indices,
    const Tensor &sorted_indices,
    const Tensor &count,
    int64_t num_weights,
    int padding_idx = -1,
    bool mode_mean = false,
    const Tensor &offset2bag = Tensor(),
    const Tensor &bag_size = Tensor(),
    const Tensor &per_sample_weights = Tensor());

}}
