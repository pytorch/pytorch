#pragma once

#include <ATen/cuda/CUDABlas.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace blas {

  template <>
  __device__ void gemm<int32>(CUDABLAS_GEMM_ARGTYPES(int32_t)) {

  }

  template <>
  __device__ void gemm<int64_t>(CUDABLAS_GEMM_ARGTYPES(int64_t)) {

  }
}
}
}
