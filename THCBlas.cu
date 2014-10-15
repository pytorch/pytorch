#include "THCBlas.h"
#include "THCGeneral.h"


void THCudaBlas_swap(long n, float *x, long incx, float *y, long incy)
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
    cublasSswap(i_n, x, i_incx, y, i_incy);
    THCublasCheck();
    return;
  }
  THError("Cublas_swap only supports n, incx and"
          " incy upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_scal(long n, float a, float *x, long incx)
{
  if(n == 1)
    incx = 1;

  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    cublasSscal(i_n, a, x, i_incx);
    THCublasCheck();
    return;
  }
  THError("Cublas_scal only supports n and incx "
          "upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_copy(long n, float *x, long incx, float *y, long incy)
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
    cublasScopy(i_n, x, i_incx, y, i_incy);
    THCublasCheck();
    return;
  }
  
  THError("Cublas_copy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
}

void THCudaBlas_axpy(long n, float a, float *x, long incx, float *y, long incy)
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
    cublasSaxpy(i_n, a, x, i_incx, y, i_incy);
    THCublasCheck();
    return;
  }

  THError("Cublas_axpy only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);

}
float THCudaBlas_dot(long n, float *x, long incx, float *y, long incy)
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
    result = cublasSdot(i_n, x, i_incx, y, i_incy);
    THCublasCheck();
    return result;
  }
  THError("Cublas_dot only supports n, incx and incy "
          "upto signed integer limits: %d", INT_MAX);
  return -1;
}

/* Level 2 */
void THCudaBlas_gemv(char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy)
{
    if(n == 1)
    lda = m;
  
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

    cublasSgemv(trans, i_m, i_n, alpha, a, i_lda, x, i_incx, beta, y, i_incy);
    THCublasCheck();
    return;
  }
  THError("Cublas_gemv only supports m, n, lda, incx, incy"
          "in the range 0 < [val] <= %d", INT_MAX);
}
void THCudaBlas_ger(long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda)
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

      cublasSger(i_m, i_n, alpha, x, i_incx, y, i_incy, a, i_lda);
      THCublasCheck();
      return;
    }
  THError("Cublas_ger only supports m, n, lda, incx, incy"
          "with the bound [val] <= %d", INT_MAX);
}

/* Level 3 */
void THCudaBlas_gemm(char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc)
{
    int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    cublasSgemm(transa, transb, i_m, i_n, i_k, alpha, a, i_lda, b, i_ldb, beta, c, i_ldc);
    THCublasCheck();
    return;
  }
  THError("Cublas_gemm only supports m, n, k, lda, ldb, ldc"
          "with the bound [val] <= %d", INT_MAX);
}
