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

/* Level 2 */

void adjustLdLevel2(int64_t m, int64_t n, int64_t *lda)
{
  // Note: leading dimensions generally are checked that they are > 0 and at least as big the result
  // requires (even if the value won't be used).
  // TODO: why does Level3 check trans but this doesn't?
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);
}

void THCudaBlas_Sger(THCState *state, int64_t m, int64_t n, float alpha, float *x, int64_t incx, float *y, int64_t incy, float *a, int64_t lda)
{
  adjustLdLevel2(m, n, &lda);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
      THCublasCheck(cublasSger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Sger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Dger(THCState *state, int64_t m, int64_t n, double alpha, double *x, int64_t incx, double *y, int64_t incy, double *a, int64_t lda)
{
  adjustLdLevel2(m, n, &lda);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;

      cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
      THCublasCheck(cublasDger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Dger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

/* Level 3 */
void THCudaBlas_Sgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, float alpha, float *a, int64_t lda, float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  at::cuda::blas::gemm<float>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// In CUDA 8.0, definition of data types for sgemmex changed
#if CUDA_VERSION < 8000
#  define CUDA_R_16F CUBLAS_DATA_HALF
#endif

void THCudaBlas_Hgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::Half alpha, at::Half *a, int64_t lda, at::Half *b, int64_t ldb, at::Half beta, at::Half *c, int64_t ldc)
{
  at::cuda::blas::gemm<at::Half>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void THCudaBlas_Bgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, at::BFloat16 alpha, at::BFloat16 *a, int64_t lda, at::BFloat16 *b, int64_t ldb, at::BFloat16 beta, at::BFloat16 *c, int64_t ldc)
{
  at::cuda::blas::gemm<at::BFloat16>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void THCudaBlas_Dgemm(THCState *state, char transa, char transb, int64_t m, int64_t n, int64_t k, double alpha, double *a, int64_t lda, double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  at::cuda::blas::gemm<double>(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
