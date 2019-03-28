#pragma once

#include <caffe2/core/types.h>

#ifdef __CUDA_ARCH__
// Proxy for including cuda_fp16.h, because common_gpu.h
// has necessary diagnostic guards.
#include <caffe2/core/common_gpu.h>
#endif
#if __HIP_DEVICE_COMPILE__
#include <caffe2/core/hip/common_gpu.h>
#endif

// See Note [hip-clang differences to hcc]

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__) || defined(__HIP__)
#define CONVERSIONS_DECL __host__ __device__ inline
#else
#define CONVERSIONS_DECL inline
#endif

namespace caffe2 {

namespace convert {

template <typename IN, typename OUT>
CONVERSIONS_DECL OUT To(const IN in) {
  return static_cast<OUT>(in);
}

template <typename OUT, typename IN>
CONVERSIONS_DECL OUT Get(IN x) {
  return static_cast<OUT>(x);
}

}; // namespace convert

}; // namespace caffe2

#undef CONVERSIONS_DECL
