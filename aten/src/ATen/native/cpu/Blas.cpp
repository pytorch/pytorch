
TH_EXTERNC void dgemv_(char *trans, int *m, int *n, double *alpha, double *a, int *lda, double *x, int *incx, double *beta, double *y, int *incy);
TH_EXTERNC void sgemv_(char *trans, int *m, int *n, float *alpha, float *a, int *lda, float *x, int *incx, float *beta, float *y, int *incy);
TH_API void THBlas_(gemv)(char trans, int64_t m, int64_t n, scalar_t alpha, scalar_t *a, int64_t lda, scalar_t *x, int64_t incx, scalar_t beta, scalar_t *y, int64_t incy);

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

static void THTensor_(addmvImpl)(THTensor *r_, THTensor *t, THTensor *mat, THTensor *vec, scalar_t beta, scalar_t alpha)
{
  if( (mat->dim() != 2) || (THTensor_nDimension(vec) != 1) )
    THError("matrix and vector expected, got %dD, %dD",
      mat->dim(), THTensor_nDimension(vec));

  if( mat->size(1) != THTensor_sizeLegacyNoScalars(vec, 0) ) {
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THDescBuff bv = THTensor_(sizeDesc)(vec);
    THError("size mismatch, %s, %s", bm.str, bv.str);
  }

  if(THTensor_nDimension(t) != 1)
    THError("vector expected, got t: %dD", t->dim());

  if(THTensor_sizeLegacyNoScalars(t, 0) != mat->size(0)) {
    THDescBuff bt = THTensor_(sizeDesc)(t);
    THDescBuff bm = THTensor_(sizeDesc)(mat);
    THError("size mismatch, t: %s, mat: %s", bt.str, bm.str);
  }

  if(r_ != t)
  {
    THTensor_(resizeAs)(r_, t);
    at::Tensor r__wrap = THTensor_wrap(r_);
    at::Tensor t_wrap = THTensor_wrap(t);
    at::native::copy_(r__wrap, t_wrap);
  }

  auto r_stride = THTensor_strideLegacyNoScalars(r_, 0);

  // n == 1 || lda >= max(1, m)
  #define LDA_COND(M, N, LDA) ((N) == 1 || (LDA) >= THMax(1, (M)))

  if(mat->stride(0) == 1 && LDA_COND(mat->size(0), mat->size(1), mat->stride(1)))
  {
    THBlas_(gemv)('n', mat->size(0), mat->size(1),
                  alpha, mat->data<scalar_t>(), mat->stride(1),
                  vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
                  beta, r_->data<scalar_t>(), r_stride);
  }
  else if(mat->stride(1) == 1 && LDA_COND(mat->size(1), mat->size(0), mat->stride(0)))
  {
    THBlas_(gemv)('t',  mat->size(1), mat->size(0),
                  alpha, mat->data<scalar_t>(), mat->stride(0),
                  vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
                  beta, r_->data<scalar_t>(), r_stride);
  }
  else
  {
    THTensor *cmat = THTensor_(newContiguous)(mat);

    THBlas_(gemv)('t',  mat->size(1), mat->size(0),
                  alpha, cmat->data<scalar_t>(), cmat->stride(0),
                  vec->data<scalar_t>(), THTensor_strideLegacyNoScalars(vec, 0),
                  beta, r_->data<scalar_t>(), r_stride);

    c10::raw::intrusive_ptr::decref(cmat);
  }

  // In gemv (x,0).mv(0) does not
  // handle beta, whereas gemm does for case where (x,0).mm(0,y).
  if (THTensor_sizeLegacyNoScalars(vec, 0) == 0 && mat->size(0) != 0) {
    if (beta == 0) {
      THTensor_(zero)(r_);
    } else if (beta != 1) {
      THTensor_(mul)(r_, r_, beta);
    }
  }

  #undef LDA_COND
}

void THTensor_(addmv)(THTensor *r_, THTensor *t, THTensor *mat, THTensor *vec, scalar_t beta, scalar_t alpha) {
  {
#ifdef BUILD_NAMEDTENSOR
    at::NoNamesGuard guard;
#endif
    THTensor_(addmvImpl)(r_, t, mat, vec, beta, alpha);
  }
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names_for_addmv(r_, mat, vec, t);
#endif
}