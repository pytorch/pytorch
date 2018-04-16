#include "THCBlas.h"
#include "THCGeneral.h"
#include "THCHalf.h"


float THCudaBlas_Sdot(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;
    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(hipblasSdot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Sdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

double THCudaBlas_Ddot(THCState *state, long n, double *x, long incx, double *y, long incy)
{
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    double result;
    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(hipblasDdot(handle, i_n, x, i_incx, y, i_incy, &result));
    return result;
  }

  THError("Cublas_Ddot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return 0;
}

#ifdef CUDA_HALF_TENSOR
half THCudaBlas_Hdot(THCState *state, long n, half *x, long incx, half *y, long incy)
{
#if CUDA_VERSION >= 8000
  if (n == 1) {
    incx = 1;
    incy = 1;
  }

  if ((n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX)) {
    half result;
#ifdef HIPBLAS_TODO
    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(hipblasDotEx(handle, n, x, CUDA_R_16F, incx, y, CUDA_R_16F, incy, &result, CUDA_R_16F, CUDA_R_32F));
#else
#ifdef __NVCC__
    cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
    cublasSetStream((cublasHandle_t)handle, THCState_getCurrentStream(state));
    cublasDotEx(handle, n, x, CUDA_R_16F, incx, y, CUDA_R_16F, incy, &result, CUDA_R_16F, CUDA_R_32F);
#endif
#endif
    return result;
   }

  THError("Cublas_Hdot only supports n, incx and incy "
          "up to signed integer limits: %d", INT_MAX);
  return THC_float2half(0);
#else
  THError("Cublas_Hdot requires CUDA 8.0+");
  return THC_float2half(0);
#endif
}
#endif

/* Level 2 */
void THCudaBlas_Sgemv(THCState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{
  if(n == 1)
    lda = m;

  hipblasOperation_t op;
  if (trans == 't') op = HIPBLAS_OP_T;
  else if (trans == 'n') op = HIPBLAS_OP_N;
  else if (trans == 'c') op = HIPBLAS_OP_C;
  else THError("Cublas_Sgemv parameter trans should be 't', 'n' or 'c'.");

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(hipblasSgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));

    return;
  }
  THError("Cublas_Sgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Dgemv(THCState *state, char trans, long m, long n, double alpha, double *a, long lda, double *x, long incx, double beta, double *y, long incy)
{
  if(n == 1)
    lda = m;

  hipblasOperation_t op;
  if (trans == 't') op = HIPBLAS_OP_T;
  else if (trans == 'n') op = HIPBLAS_OP_N;
  else if (trans == 'c') op = HIPBLAS_OP_C;
  else THError("Cublas_Sgemv parameter trans should be 't', 'n' or 'c'.");

  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(hipblasDgemv(handle, op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_Dgemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_Sger(THCState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;
      hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
      hipblasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(hipblasSger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_Sger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_Dger(THCState *state, long m, long n, double alpha, double *x, long incx, double *y, long incy, double *a, long lda)
{
  if(n == 1)
    lda = m;

  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
    {
      int i_m = (int)m;
      int i_n = (int)n;
      int i_lda = (int)lda;
      int i_incx = (int)incx;
      int i_incy = (int)incy;
#ifdef HIPBLAS_TODO
      hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
      hipblasSetStream(handle, THCState_getCurrentStream(state));
      THCublasCheck(hipblasDger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
#else
#ifdef __NVCC__
    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream((hipblasHandle_t)handle, THCState_getCurrentStream(state));
    hipblasDger(handle, i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda);
#endif
#endif
      return;
    }
  THError("Cublas_Dger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

hipblasOperation_t convertTransToHipblasOperation(char trans) {
  if (trans == 't') return HIPBLAS_OP_T;
  else if (trans == 'n') return HIPBLAS_OP_N;
  else if (trans == 'c') return HIPBLAS_OP_C;
  else {
    THError("trans must be one of: t, n, c");
    return HIPBLAS_OP_T;
  }
}
#ifdef __NVCC__
cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    THError("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}
#endif
void adjustLd(char transa, char transb, long m, long n, long k, long *lda, long *ldb, long *ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    *ldc = m;

  if(transa_)
  {
    if(m == 1)
      *lda = k;
  }
  else
  {
    if(k == 1)
      *lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

/* Level 3 */
void THCudaBlas_Sgemm(THCState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  hipblasOperation_t opa = convertTransToHipblasOperation(transa);
  hipblasOperation_t opb = convertTransToHipblasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(hipblasSgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Sgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

#ifdef CUDA_HALF_TENSOR
// In CUDA 8.0, definition of data types for sgemmex changed
#if CUDA_VERSION < 8000
#  define CUDA_R_16F CUBLAS_DATA_HALF
#endif

void THCudaBlas_Hgemm(THCState *state, char transa, char transb, long m, long n, long k, half alpha, half *a, long lda, half *b, long ldb, half beta, half *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  hipblasOperation_t opa = convertTransToHipblasOperation(transa);
  hipblasOperation_t opb = convertTransToHipblasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));

    // Check for native Hgemm support
#ifdef __NVCC__
    if (THC_fastHalfInstructions(state)) {
      THCublasCheck(hipblasHgemm(handle, opa, opb,
				i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb,
				&beta, c, i_ldc));
    } else {
      // Simulated Hgemm
      float fAlpha = THC_half2float(alpha);
      float fBeta = THC_half2float(beta);

      /*THCublasCheck(hipblasSgemmEx(handle, opa, opb,
				  i_m, i_n, i_k, &fAlpha,
                                  a, CUDA_R_16F, i_lda, b, CUDA_R_16F,
				  i_ldb, &fBeta, c, CUDA_R_16F, i_ldc));*/
    }
#elif __HCC__
//       hipblasHgemm(handle, opa, opb,
// 				i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb,
// 				&beta, c, i_ldc);
#endif
    return;
  }
  THError("Cublas_Hgemm only supports m, n, k, lda, ldb, ldc"
          "with th bound [val] <= %d", INT_MAX);

}
#endif

void THCudaBlas_Dgemm(THCState *state, char transa, char transb, long m, long n, long k, double alpha, double *a, long lda, double *b, long ldb, double beta, double *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  hipblasOperation_t opa = convertTransToHipblasOperation(transa);
  hipblasOperation_t opb = convertTransToHipblasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
    hipblasSetStream(handle, THCState_getCurrentStream(state));
    THCublasCheck(hipblasDgemm(handle, opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_Dgemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

#ifdef __NVCC__
#if CUDA_VERSION >= 9010
void THCudaBlas_HgemmStridedBatched(THCState *state, char transa, char transb, long m, long n, long k,
                             half alpha, const half *a, long lda, long strideA, const half *b, long ldb, long strideB,
                             half beta, half *c, long ldc, long strideC, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )

  {
    THError("Cublas_SgemmStridedBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  cublasHandle_t handle = THCState_getCurrentBlasHandle(state);
  cublasSetStream(handle, THCState_getCurrentStream(state));
  float fAlpha = THC_half2float(alpha);
  float fBeta = THC_half2float(beta);
  THCublasCheck(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
  THCublasCheck(cublasGemmStridedBatchedEx(handle,
                                   opa, opb, (int)m, (int)n, (int)k,
                                   (void*)&fAlpha, a, CUDA_R_16F, (int)lda, strideA,
                                   b, CUDA_R_16F, (int)ldb, strideB,
                                   (void*)&fBeta, c, CUDA_R_16F, (int)ldc, strideC,
                                   (int)batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  THCublasCheck(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}
#endif
#endif

void THCudaBlas_SgemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                             float alpha, const float *a[], long lda, const float *b[], long ldb,
                             float beta, float *c[], long ldc, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_SgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  hipblasOperation_t opa = convertTransToHipblasOperation(transa);
  hipblasOperation_t opb = convertTransToHipblasOperation(transb);

  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  for (long i = 0; i < batchCount; i++) {
    THCublasCheck(hipblasSgemm(handle,
                              opa, opb, (int)m, (int)n, (int)k,
                              &alpha, a[i], (int)lda, b[i], (int)ldb, &beta, c[i], (int)ldc));
  }
}

void THCudaBlas_DgemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                             double alpha, const double *a[], long lda, const double *b[], long ldb,
                             double beta, double *c[], long ldc, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_DgemmBatched only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  hipblasOperation_t opa = convertTransToHipblasOperation(transa);
  hipblasOperation_t opb = convertTransToHipblasOperation(transb);

  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  for (long i = 0; i < batchCount; i++) {
    THCublasCheck(hipblasDgemm(handle,
                              opa, opb, (int)m, (int)n, (int)k,
                              &alpha, a[i], (int)lda, b[i], (int)ldb, &beta, c[i], (int)ldc));
  }
}

/* Inverse */
void THCudaBlas_Sgetrf(THCState *state, int n, float **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Sgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
#ifdef HIPBLAS_TODO
  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(hipblasSgetrfBatched(handle, n, a, lda, pivot, info, batchSize));
#endif
}

void THCudaBlas_Dgetrf(THCState *state, int n, double **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
#ifdef HIPBLAS_TODO
  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(hipblasDgetrfBatched(handle, n, a, lda, pivot, info, batchSize));
#endif
}

THC_API void THCudaBlas_Sgetrs(THCState *state, char transa, int n, int nrhs, const float **a, int lda, int *pivot, float **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }


#ifdef HIPBLAS_TODO
  hipblasOperation_t opa = convertTransToHipblasOperation(transa);
  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(hipblasSgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
#endif
}


THC_API void THCudaBlas_Dgetrs(THCState *state, char transa, int n, int nrhs, const double **a, int lda, int *pivot, double **b, int ldb, int *info, int batchSize)
{
  if( (n >= INT_MAX) || (nrhs >= INT_MAX) || (lda >= INT_MAX) || (ldb >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetrs only supports n, nrhs, lda, ldb, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }


#ifdef HIPBLAS_TODO
  hipblasOperation_t opa = convertTransToHipblasOperation(transa);
  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(hipblasDgetrsBatched(handle, opa, n, nrhs, a, lda, pivot, b, ldb, info, batchSize));
#endif
}
void THCudaBlas_Sgetri(THCState *state, int n, const float **a, int lda, int *pivot, float **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Sgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
#ifdef HIPBLAS_TODO
  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(hipblasSgetriBatched(handle, n, a, lda, pivot, c, ldc, info, batchSize));
#endif
}

void THCudaBlas_Dgetri(THCState *state, int n, const double **a, int lda, int *pivot, double **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_Dgetri only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
#ifdef HIPBLAS_TODO
  hipblasHandle_t handle = THCState_getCurrentBlasHandle(state);
  hipblasSetStream(handle, THCState_getCurrentStream(state));
  THCublasCheck(hipblasDgetriBatched(handle, n, a, lda, pivot, c, ldc, info, batchSize));
#endif
}
