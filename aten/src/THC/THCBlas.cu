#include <THC/THCBlas.h>
#include <THC/THCGeneral.h>
#include <TH/THHalf.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDABlas.h>

#include <algorithm>
#include <mutex>

#ifdef __HIP_PLATFORM_HCC__
#include <hip/hip_version.h>
#endif

/* Level 3 */
void THCudaBlas_Sgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  at::cuda::blas::gemm<float>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void THCudaBlas_Hgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::Half alpha, at::Half *a, int64_t lda, at::Half *b, int64_t ldb, at::Half beta, at::Half *c, int64_t ldc)
{
  at::cuda::blas::gemm<at::Half>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#ifdef __HIP_PLATFORM_HCC__
void THCudaBlas_Bgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::BFloat16 alpha, at::BFloat16 *a, int64_t lda, at::BFloat16 *b, int64_t ldb, at::BFloat16 beta, at::BFloat16 *c, int64_t ldc)
{
  at::cuda::blas::gemm<at::BFloat16>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
#endif

void THCudaBlas_Dgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  at::cuda::blas::gemm<double>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
