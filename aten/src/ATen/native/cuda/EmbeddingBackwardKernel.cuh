#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>

#include <ATen/AccumulateType.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCAtomics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>

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
    bool scale_grad_by_freq = false,
    bool mode_mean = false,
    const Tensor &offset2bag = Tensor(),
    const Tensor &bag_size = Tensor(),
    const Tensor &per_sample_weights = Tensor());

}}
