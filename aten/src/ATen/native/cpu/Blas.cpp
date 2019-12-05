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