#include <ATen/cuda/CUDABlas.h>
#include <c10/macros/Macros.h>

namespace at {
namespace cuda {
namespace blas {

  template <typename scalar_t>
  __global__ void gemm_kernel(CUDABLAS_GEMM_ARGTYPES(scalar_t)) {
  }

  template <>
  void gemm<int32_t>(CUDABLAS_GEMM_ARGTYPES(int32_t)) {
    constexpr int64_t block_size = 256;
    const int64_t grid = 1;
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (transa == 'n' && transb == 'n') {
      gemm_kernel<<<grid, block_size,0,stream>>>(transa, transb, m, n, k, alpha,
                                                 a, lda, b, ldb, beta, c, ldc);
    }
  }

  template <>
  void gemm<int64_t>(CUDABLAS_GEMM_ARGTYPES(int64_t)) {
    constexpr int64_t block_size = 256;
    const int64_t grid = 1;
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (transa == 'n' && transb == 'n') {
      gemm_kernel<<<grid, block_size,0,stream>>>(transa, transb, m, n, k, alpha,
                                                 a, lda, b, ldb, beta, c, ldc);
    }
  }

}
}
}
