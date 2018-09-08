#pragma once

#include "ATen/cuda/ATenCUDAGeneral.h"
#include "ATen/core/Half.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace at {

template <> AT_CUDA_API
struct Converter<half, Half> {
  half operator()(Half);
};

template <> AT_CUDA_API
struct Converter<Half, half> {
  Half operator()(half);
};

template <> AT_CUDA_API
struct Converter<half, double> {
  half operator()(double);
};

#if CUDA_VERSION >= 9000 || defined(__HIP_PLATFORM_HCC__)
template <> __half HalfFix(Half h);
template <> Half HalfFix(__half h);
#endif
} // namespace at
