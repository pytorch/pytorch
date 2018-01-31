#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZBlas.c"
#else

#include "THZTypeMacros.h"

// extern blas header only supports float numbers
#if defined(THZ_NTYPE_IS_COMPLEX)
TH_EXTERNC void THZ_BLAS_NAME(swap)(int *n, ntype *x, int *incx, ntype *y, int *incy);
TH_EXTERNC void THZ_BLAS_NAME(scal)(int *n, ntype *a, ntype *x, int *incx);
TH_EXTERNC void THZ_BLAS_NAME(copy)(int *n, ntype *x, int *incx, ntype *y, int *incy);
TH_EXTERNC void THZ_BLAS_NAME(axpy)(int *n, ntype *a, ntype *x, int *incx, ntype *y, int *incy);
TH_EXTERNC void THZ_BLAS_NAME(gemv)(char *trans, int *m, int *n, ntype *alpha, ntype *a, int *lda, ntype *x, int *incx, ntype *beta, ntype *y, int *incy);
TH_EXTERNC void THZ_BLAS_NAME(gemm)(char *transa, char *transb, int *m, int *n, int *k, ntype *alpha, ntype *a, int *lda, ntype *b, int *ldb, ntype *beta, ntype *c, int *ldc);


// begin dot
/* blas.dot declaration
 * cblas is re-wrapped to fortran form
 * for complex type, dot use dotc rather
 * than complex dot
 */
TH_EXTERNC ntype THZ_BLAS_NAME(dotc)(int *n, ntype *x, int *incx, ntype *y, int *incy);

// begin ger

/*
 * ger declaration
 * complex will use gerc
 */

TH_EXTERNC void THZ_BLAS_NAME(ger)(int *m, int *n, ntype *alpha, ntype *x, int *incx, ntype *y, int *incy, ntype *a, int *lda);
TH_EXTERNC void THZ_BLAS_NAME(gerc)(int *m, int *n, ntype *alpha, ntype *x, int *incx, ntype *y, int *incy, ntype *a, int *lda);

#endif // THZ_NTYPE_IS_COMPLEX

void THZBlas_(swap)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    THZ_BLAS_NAME(swap)(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
    {
      ntype z = x[i*incx];
      x[i*incx] = y[i*incy];
      y[i*incy] = z;
    }
  }
}

void THZBlas_(scal)(int64_t n, ntype a, ntype *x, int64_t incx)
{
  if(n == 1)
    incx = 1;

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;

    THZ_BLAS_NAME(scal)(&i_n, &a, x, &i_incx);
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

void THZBlas_(copy)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    THZ_BLAS_NAME(copy)(&i_n, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] = x[i*incx];
  }
}

void THZBlas_(axpy)(int64_t n, ntype a, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    THZ_BLAS_NAME(axpy)(&i_n, &a, x, &i_incx, y, &i_incy);
    return;
  }
#endif
  {
    int64_t i;
    for(i = 0; i < n; i++)
      y[i*incy] += a*x[i*incx];
  }
}

ntype THZBlas_(dot)(int64_t n, ntype *x, int64_t incx, ntype *y, int64_t incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    return (ntype) THZ_BLAS_NAME(dotc)(&i_n, x, &i_incx, y, &i_incy);
  }
#endif
  {
    int64_t i;
    ntype sum = 0;
    for(i = 0; i < n; i++)
    sum += x[i*incx]*y[i*incy];
    return sum;
  }
}

void THZBlas_(gemv)(char trans, int64_t m, int64_t n, ntype alpha, ntype *a, int64_t lda, ntype *x, int64_t incx, ntype beta, ntype *y, int64_t incy)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda >= THMax(1, m)) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    THZ_BLAS_NAME(gemv)(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
    return;
  }
#endif
  {
    int64_t i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        ntype sum = 0;
        ntype *row_ = a+lda*i;
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
        THZBlas_(scal)(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        ntype *column_ = a+lda*j;
        ntype z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}

void THZBlas_(ger)(int64_t m, int64_t n, ntype alpha, ntype *x, int64_t incx, ntype *y, int64_t incy, ntype *a, int64_t lda)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda >= THMax(1, m)) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    THZ_BLAS_NAME(gerc)(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
    return;
  }
#endif
  {
    int64_t i, j;
    for(j = 0; j < n; j++)
    {
      ntype *column_ = a+j*lda;
      ntype z = alpha*y[j*incy];
      for(i = 0; i < m; i++)
        column_[i] += z*x[i*incx] ;
    }
  }
}

void THZBlas_(gemm)(char transa, char transb, int64_t m, int64_t n, int64_t k, ntype alpha, ntype *a, int64_t lda, ntype *b, int64_t ldb, ntype beta, ntype *c, int64_t ldc)
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

#if defined(USE_BLAS) && defined(THZ_NTYPE_IS_FPOINT)
  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) &&
      (lda >= THMax(1, (transa_ ? k : m))) && (lda <= INT_MAX) &&
      (ldb >= THMax(1, (transb_ ? n : k))) && (ldb <= INT_MAX) &&
      (ldc >= THMax(1, m)) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

    THZ_BLAS_NAME(gemm)(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
    return;
  }
#endif
  {
    int64_t i, j, l;
    if(!transa_ && !transb_)
    {
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
    if (beta == 0)
      c[j*ldc+i] = alpha*sum;
    else
      c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
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
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
    if (beta == 0)
      c[j*ldc+i] = alpha*sum;
    else
      c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else
    {
      ntype *a_ = a;
      for(i = 0; i < m; i++)
      {
        ntype *b_ = b;
        for(j = 0; j < n; j++)
        {
          ntype sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
    if (beta == 0)
      c[j*ldc+i] = alpha*sum;
    else
      c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
  }
}

#endif
