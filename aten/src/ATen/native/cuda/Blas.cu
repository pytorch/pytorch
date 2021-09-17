#include <ATen/cuda/CUDABlas.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/Loops.cuh>
#include <c10/macros/Macros.h>

#include <iostream>

namespace at {
namespace cuda {
namespace blas {

  template <typename scalar_t, bool transA=false, bool transB=false>
  __global__ void gemm_kernel(int64_t M, int64_t N, int64_t K, scalar_t alpha,
                                                  const scalar_t* a, int64_t lda, const scalar_t * b, int64_t ldb,
                                                  scalar_t beta, scalar_t *c, int64_t ldc) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    scalar_t product = 0;

    if (row < M && col < N) {
      for (int64_t k = 0; k < K; ++k) {
        if (transA == false && transB == false) {
          product += a[row + k * lda] * b[k  + col * ldb];
        }
        else if (transA == true && transB == false) {
          product += a[k + row * lda] * b[k + col * ldb];
        }
        else if (transA == false && transB == true) {
          product += a[row + k * lda] * b[col + k * ldb];
        }
        else {
          product += a[k + row * lda] * b[col  + k * ldb];
        }
      }
      c[row + col * ldc] = beta * c[row + col * ldc] + alpha * product;
    }

  }

  template <typename scalar_t>
  void launch_gemm_kernel(CUDABLAS_GEMM_ARGTYPES(scalar_t))  {
    uint32_t thread_block_dim = std::sqrt(at::cuda::getApplyBlockSize());
    const dim3 thread_block(thread_block_dim, thread_block_dim, 1);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(std::ceil(double(m)/double(thread_block_dim)) + 1, std::ceil(double(n)/double(thread_block_dim)) + 1, 1);

    // if-else here to avoid putting it in the critical path in the kernel.
    if (transa == 'n' && transb == 'n') {
      gemm_kernel<scalar_t, false, false><<<grid, thread_block, 0, stream>>>
        (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else if (transa == 't' && transb == 'n') {
      gemm_kernel<scalar_t, true, false><<<grid, thread_block, 0, stream>>>
        (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else if (transa == 'n' && transb == 't') {
      gemm_kernel<scalar_t, false, true><<<grid, thread_block, 0, stream>>>
        (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    else {
      gemm_kernel<scalar_t, true, true><<<grid, thread_block, 0, stream>>>
        (m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  template <>
  void integer_gemm<int32_t>(CUDABLAS_GEMM_ARGTYPES(int32_t)) {
    launch_gemm_kernel(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }

  template <>
  void integer_gemm<int64_t>(CUDABLAS_GEMM_ARGTYPES(int64_t)) {
    launch_gemm_kernel(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}
}
}
