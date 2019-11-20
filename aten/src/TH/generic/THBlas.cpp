#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THBlas.cpp"
#else

#ifdef USE_FBGEMM
#include "fbgemm/FbgemmI64.h"
#endif // USE_FBGEMM

#ifdef BLAS_F2C
# define ffloat double
#else
# define ffloat float
#endif

TH_EXTERNC void dswap_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void sswap_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void dscal_(int *n, double *a, double *x, int *incx);
TH_EXTERNC void sscal_(int *n, float *a, float *x, int *incx);
TH_EXTERNC void dcopy_(int *n, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void scopy_(int *n, float *x, int *incx, float *y, int *incy);
TH_EXTERNC void daxpy_(int *n, double *a, double *x, int *incx, double *y, int *incy);
TH_EXTERNC void saxpy_(int *n, float *a, float *x, int *incx, float *y, int *incy);
TH_EXTERNC double ddot_(int *n, double *x, int *incx, double *y, int *incy);
#ifdef BLAS_USE_CBLAS_DOT
TH_EXTERNC float cblas_sdot(const int n, const float *x, const int incx, const float *y, const int incy);
#ifndef THBlas_C_sdot_
#define THBlas_C_sdot_
static inline ffloat sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy)
{
  return cblas_sdot(*n, x, *incx, y, *incy);
}
#endif
#else
TH_EXTERNC ffloat sdot_(int *n, float *x, int *incx, float *y, int *incy);
#endif
TH_EXTERNC void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
TH_EXTERNC void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
TH_EXTERNC void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda);
TH_EXTERNC void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
TH_EXTERNC void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb, double *beta, double *c, int *ldc);
TH_EXTERNC void sgemm_(char *transa, char *transb, int *m, int *n, int *k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);



void THBlas_(swap)(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dswap_(&i_n, x, &i_incx, y, &i_incy);
#else
    sswap_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
    {
      scalar_t z = x[i*incx];
      x[i*incx] = y[i*incy];
      y[i*incy] = z;
    }
  }
}

void THBlas_(scal)(int64_t n, scalar_t a, scalar_t *x, int64_t incx)
{
  if(n == 1)
    incx = 1;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;

#if defined(TH_REAL_IS_DOUBLE)
    dscal_(&i_n, &a, x, &i_incx);
#else
    sscal_(&i_n, &a, x, &i_incx);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++) {
      if (a == 0) {
        x[i*incx] = 0;
      } else {
        x[i*incx] *= a;
      }
    }
  }
}

void THBlas_(copy)(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dcopy_(&i_n, x, &i_incx, y, &i_incy);
#else
    scopy_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] = x[i*incx];
  }
}

void THBlas_(axpy)(int64_t n, scalar_t a, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    daxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#else
    saxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] += a*x[i*incx];
  }
}

scalar_t THBlas_(dot)(int64_t n, scalar_t *x, int64_t incx, scalar_t *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    return (scalar_t) ddot_(&i_n, x, &i_incx, y, &i_incy);
#else
    return (scalar_t) sdot_(&i_n, x, &i_incx, y, &i_incy);
#endif
  }
#endif
  {
    int64_t i;
    scalar_t sum = 0;
    for(i = 0; i < n; i++)
    sum += x[i*incx]*y[i*incy];
    return sum;
  }
}

void THBlas_(gemv)(
  char trans,
  int64_t m,
  int64_t n,
  scalar_t alpha,
  scalar_t *a,
  int64_t lda,
  scalar_t *x,
  int64_t incx,
  scalar_t beta,
  scalar_t *y,
  int64_t incy)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    THArgCheck(lda >= THMax(1, m), 6,
      "lda should be at least max(1, m=%d), but have %d", m, lda);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#else
    sgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#endif
    return;
  }
#endif
  {
    int64_t i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        scalar_t sum = 0;
        scalar_t *row_ = a+lda*i;
        for(j = 0; j < m; j++)
          sum += x[j*incx]*row_[j];
          if (beta == 0)
            y[i*incy] = alpha*sum;
          else
            y[i*incy] = beta*y[i*incy] + alpha*sum;
      }
    }
    else
    {
      if(beta != 1)
        THBlas_(scal)(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        scalar_t *column_ = a+lda*j;
        scalar_t z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}

void THBlas_(ger)(
  int64_t m,
  int64_t n,
  scalar_t alpha,
  scalar_t *x,
  int64_t incx,
  scalar_t *y,
  int64_t incy,
  scalar_t *a,
  int64_t lda)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    THArgCheck(lda >= THMax(1, m), 9,
      "lda should be at least max(1, m=%d), but have %d", m, lda);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(TH_REAL_IS_DOUBLE)
    dger_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#else
    sger_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#endif
    return;
  }
#endif
  {
    int64_t i, j;
    for(j = 0; j < n; j++)
    {
      scalar_t *column_ = a+j*lda;
      scalar_t z = alpha*y[j*incy];
      for(i = 0; i < m; i++)
        column_[i] += z*x[i*incx] ;
    }
  }
}

void THBlas_(gemm)(
  char transa,
  char transb,
  int64_t m,
  int64_t n,
  int64_t k,
  scalar_t alpha,
  scalar_t *a,
  int64_t lda,
  scalar_t *b,
  int64_t ldb,
  scalar_t beta,
  scalar_t *c,
  int64_t ldc)
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

#if defined(USE_BLAS) && (defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) &&
      (lda <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    THArgCheck(lda >= THMax(1, (transa_ ? k : m)), 8,
      "lda should be at least max(1, %d), but have %d", (transa_ ? k : m), lda);
    THArgCheck(ldb >= THMax(1, (transb_ ? n : k)), 10,
      "ldb should be at least max(1, %d), but have %d", (transb_ ? n : k), ldb);
    THArgCheck(ldc >= THMax(1, m), 13,
      "ldc should be at least max(1, m=%d), but have %d", m, ldc);
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

#if defined(TH_REAL_IS_DOUBLE)
    dgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#else
    sgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#endif
    return;
  }
#endif

#if defined(USE_FBGEMM) && defined(TH_REAL_IS_LONG)
  if (alpha == 1 && (beta == 0 || beta == 1)) {
    // In FBGEMM, we assume row-major ordering; However, here we assume the
    // column-major ordering following the FORTRAN tradition in BLAS interface
    // in this function: we can configure the layout (row/column-major ordering)
    // of A and B by changing transa_ and transb_, but we cannot change the
    // layout of C with this FORTRAN-style BLAS interface.
    //
    // The workaround is that we compute
    // C^T (n x m) = B^T (n x k) * A^T (k x m) instead.
    //
    // In this way we view C^T as the row-major ordering when passing to FBGEMM.
    fbgemm::cblas_gemm_i64_i64acc(
        transb_ ? fbgemm::matrix_op_t::Transpose
                : fbgemm::matrix_op_t::NoTranspose,
        transa_ ? fbgemm::matrix_op_t::Transpose
                : fbgemm::matrix_op_t::NoTranspose,
        n,
        m,
        k,
        b,
        ldb,
        a,
        lda,
        beta == 1,
        c,
        ldc);
    return;
  }
#endif

  {
    if(!transa_ && !transb_)
    {
      if (beta == 0) {
        for (int64_t j = 0; j < n; j++) {
          for (int64_t i = 0; i < m; i++) {
            c[j * ldc + i] = 0;
          }
        }
      }
      else {
        for (int64_t j = 0; j < n; j++) {
          for (int64_t i = 0; i < m; i++) {
            c[j * ldc + i] *= beta;
          }
        }
      }
      for (int64_t l = 0; l < k; l++) {
        for (int64_t j = 0; j < n; j++) {
          scalar_t val = b[l + j * ldb] * alpha;
          int64_t i_m = m / 4;
          for (int64_t i_i = 0; i_i < i_m; i_i++) {
              c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
              c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
              c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
              c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
          }
          int64_t i = i_m * 4;
          for (; i < m; i++)
              c[j * ldc + i] += a[i + l * lda] * val;
        }
      }
    }
    else if(transa_ && !transb_)
    {
      int64_t i, j, l;
      scalar_t *a_ = a;
      for(i = 0; i < m; i++)
      {
        scalar_t *b_ = b;
        for(j = 0; j < n; j++)
        {
          scalar_t sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
          if (beta == 0)
            c[j*ldc+i] = alpha*sum;
          else
            c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
      if (beta == 0) {
        for (int64_t j = 0; j < n; j++) {
          for (int64_t i = 0; i < m; i++) {
            c[j * ldc + i] = 0;
          }
        }
      }
      else {
        for (int64_t j = 0; j < n; j++) {
          for (int64_t i = 0; i < m; i++) {
            c[j * ldc + i] *= beta;
          }
        }
      }
      for (int64_t l = 0; l < k; l++) {
        for (int64_t j = 0; j < n; j++) {
          scalar_t val = b[j + l * ldb] * alpha;
          int64_t i_m = m / 4;
          for (int64_t i_i = 0; i_i < i_m; i_i++) {
            c[j * ldc + i_i * 4 + 0] += a[i_i * 4 + 0 + l * lda] * val;
            c[j * ldc + i_i * 4 + 1] += a[i_i * 4 + 1 + l * lda] * val;
            c[j * ldc + i_i * 4 + 2] += a[i_i * 4 + 2 + l * lda] * val;
            c[j * ldc + i_i * 4 + 3] += a[i_i * 4 + 3 + l * lda] * val;
          }
          int64_t i = i_m * 4;
          for (; i < m; i++)
            c[j * ldc + i] += a[i + l * lda] * val;
        }
      }
    }
    else
    {
      for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
          if (beta == 0)
            c[j * ldc + i] = 0;
          else
            c[j * ldc + i] *= beta;
        }
      }
      for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
          int64_t l_k = k / 4;
          for (int64_t l_l = 0; l_l < l_k; l_l++) {
              c[j * ldc + i] += a[i * lda + l_l * 4 + 0] //
                          * b[(l_l * 4 + 0) * ldb + j] * alpha;
              c[j * ldc + i] += a[i * lda + l_l * 4 + 1] //
                          * b[(l_l * 4 + 1) * ldb + j] * alpha;
              c[j * ldc + i] += a[i * lda + l_l * 4 + 2] //
                          * b[(l_l * 4 + 2) * ldb + j] * alpha;
              c[j * ldc + i] += a[i * lda + l_l * 4 + 3] //
                          * b[(l_l * 4 + 3) * ldb + j] * alpha;
          }
          int64_t l = l_k * 4;
          for (; l < k; l++)
              c[j * ldc + i] += a[i * lda + l] * b[l * ldb + j] * alpha;
        }
      }
    }
  }
}

#endif
