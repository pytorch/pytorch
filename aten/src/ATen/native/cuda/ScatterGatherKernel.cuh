#pragma once
#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCApply.cuh>

#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace {
  template <typename index_t, typename scalar_t, int dims>
#ifdef __HIP_PLATFORM_HCC__
  C10_LAUNCH_BOUNDS_1(512)
#endif
  __global__ void scatter_kernel(
                                 TensorInfo<scalar_t, index_t> tensor,
                                 TensorInfo<scalar_t, index_t> src,
                                 TensorInfo<int64_t, index_t> index,
                                 const int dim,
                                 const index_t total_elements);
}
