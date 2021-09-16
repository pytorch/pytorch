#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <c10/macros/Macros.h>

namespace at {
namespace cuda {
namespace blas {

  template <typename scalar_t>
  __global__ void gemm_kernel(int64_t M, int64_t N, int64_t K, scalar_t alpha, const scalar_t* a, int64_t lda,
                              const scalar_t * b, int64_t ldb, scalar_t beta, scalar_t *c,
                              int64_t ldc) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {

    }
  }

  template <typename scalar_t>
  void launch_gemm_kernel(CUDABLAS_GEMM_ARGTYPES(scalar_t))  {
    uint32_t thread_block_dim = std::sqrt(at::cuda::getApplyBlockSize());
    const dim3 thread_block(thread_block_dim, thread_block_dim, 1);
    const dim3 grid(std::ceil(m/thread_block_dim), std::ceil(n/thread_block_dim), 1);
    const auto stream = at::cuda::getCurrentCUDAStream();

    if (transa == 'n' && transb == 'n') {
      gemm_kernel<<<grid, thread_block, 0, stream>>>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }

  template <>
  void gemm<int32_t>(CUDABLAS_GEMM_ARGTYPES(int32_t)) {
    launch_gemm_kernel(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template <>
  void gemm<int64_t>(CUDABLAS_GEMM_ARGTYPES(int64_t)) {
    launch_gemm_kernel(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template <>
  void batched_integer_gemm<int32_t>(TensorIterator& iter, char transa, char transb) {
  }

  template <>
  void batched_integer_gemm<int64_t>(TensorIterator& iter, char transa, char transb) {
  }

}
}
}
