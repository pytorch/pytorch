#include "THCBlas.h"
#include "THCGeneral.h"

void THCudaBlas_swap(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    THCublasCheck(cublasSswap(THCState_getCurrentBlasHandle(state), i_n, x, i_incx, y, i_incy));
    return;
  }
  THError("Cublas_swap only supports n, incx and"
          " incy upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_scal(THCState *state, long n, float a, float *x, long incx)
{
  if(n == 1)
    incx = 1;

  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    THCublasCheck(cublasSscal(THCState_getCurrentBlasHandle(state), i_n, &a, x, i_incx));
    return;
  }
  THError("Cublas_scal only supports n and incx "
          "upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_copy(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    THCublasCheck(cublasScopy(THCState_getCurrentBlasHandle(state), i_n, x, i_incx, y, i_incy));
    return;
  }

  THError("Cublas_copy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_axpy(THCState *state, long n, float a, float *x, long incx, float *y, long incy)
{
    if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    THCublasCheck(cublasSaxpy(THCState_getCurrentBlasHandle(state), i_n, &a, x, i_incx, y, i_incy));
    return;
  }

  THError("Cublas_axpy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
}

float THCudaBlas_dot(THCState *state, long n, float *x, long incx, float *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;
    float result;
    THCublasCheck(cublasSdot(THCState_getCurrentBlasHandle(state), i_n, x, i_incx, y, i_incy, &result));
    cudaDeviceSynchronize();
    return result;
  }
  THError("Cublas_dot only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
  return -1;
}

/* Level 2 */
void THCudaBlas_gemv(THCState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{
  if(n == 1)
    lda = m;

  cublasOperation_t op;
  if (trans == 't') op = CUBLAS_OP_T;
  else if (trans == 'n') op = CUBLAS_OP_N;
  else if (trans == 'c') op = CUBLAS_OP_C;

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

    THCublasCheck(cublasSgemv(THCState_getCurrentBlasHandle(state), op, i_m, i_n, &alpha, a, i_lda, x, i_incx, &beta, y, i_incy));
    return;
  }
  THError("Cublas_gemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}

void THCudaBlas_ger(THCState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
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

      THCublasCheck(cublasSger(THCState_getCurrentBlasHandle(state), i_m, i_n, &alpha, x, i_incx, y, i_incy, a, i_lda));
      return;
    }
  THError("Cublas_ger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

cublasOperation_t convertTransToCublasOperation(char trans) {
  if (trans == 't') return CUBLAS_OP_T;
  else if (trans == 'n') return CUBLAS_OP_N;
  else if (trans == 'c') return CUBLAS_OP_C;
  else {
    THError("trans must be one of: t, n, c");
    return CUBLAS_OP_T;
  }
}

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
void THCudaBlas_gemm(THCState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    THCublasCheck(cublasSgemm(THCState_getCurrentBlasHandle(state), opa, opb, i_m, i_n, i_k, &alpha, a, i_lda, b, i_ldb, &beta, c, i_ldc));
    return;
  }
  THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}

void THCudaBlas_gemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                            float alpha, const float *a[], long lda, const float *b[], long ldb,
                            float beta, float *c[], long ldc, long batchCount)
{
  if( (m >= INT_MAX) || (n >= INT_MAX) || (k >= INT_MAX) || (lda >= INT_MAX)  || (ldb >= INT_MAX) || (ldc >= INT_MAX) || (batchCount >= INT_MAX) )
  {
    THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc, batchCount"
            "with the bound [val] <= %d", INT_MAX);
  }

  adjustLd(transa, transb, m, n, k, &lda, &ldb, &ldc);
  cublasOperation_t opa = convertTransToCublasOperation(transa);
  cublasOperation_t opb = convertTransToCublasOperation(transb);

  THCublasCheck(cublasSgemmBatched(THCState_getCurrentBlasHandle(state),
                                   opa, opb, (int)m, (int)n, (int)k,
                                   &alpha, a, (int)lda, b, (int)ldb, &beta, c, (int)ldc,
                                   (int)batchCount));
}

/* Inverse */
void THCudaBlas_getrf(THCState *state, int n, float **a, int lda, int *pivot, int *info, int batchSize) {
  if( (n >= INT_MAX) || (lda >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_getrf only supports n, lda, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  THCublasCheck(cublasSgetrfBatched(THCState_getCurrentBlasHandle(state), n, a, lda, pivot, info, batchSize));
}

void THCudaBlas_getri(THCState *state, int n, const float **a, int lda, int *pivot, float **c, int ldc, int *info, int batchSize) {

  if( (n >= INT_MAX) || (lda >= INT_MAX)|| (ldc >= INT_MAX) || (batchSize >= INT_MAX) )
  {
    THError("Cublas_getrf only supports n, lda, ldc, batchSize"
            "with the bound [val] <= %d", INT_MAX);
  }
  THCublasCheck(cublasSgetriBatched(THCState_getCurrentBlasHandle(state), n, a, lda, pivot, c, ldc, info, batchSize));
}
