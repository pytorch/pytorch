#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathMagma.cu"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

THC_API void THCTensor_(gesv)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(b_->nDimension == 2, 2, "b should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 1, "A should be square");
  THArgCheck(b_->size[0] == a_->size[0], 2, "A,b size incompatible");

  int n = a_->size[0];
  int nrhs = b_->size[1];

  THCTensor *a = THCTensor_(newColumnMajor)(state, ra_, a_);
  THCTensor *b = THCTensor_(newColumnMajor)(state, rb_, b_);
  real *a_data = THCTensor_(data)(state, a);
  real *b_data = THCTensor_(data)(state, b);

  int *ipiv = th_magma_malloc_pinned<int>(n);

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_sgesv_gpu(n, nrhs, a_data, n, ipiv, b_data, n, &info);
#else
  magma_dgesv_gpu(n, nrhs, a_data, n, ipiv, b_data, n, &info);
#endif

  if (info < 0)
    THError("MAGMA gesv : Argument %d : illegal value", -info);
  else if (info > 0)
    THError("MAGMA gesv : U(%d,%d) is zero, singular U.", info, info);

  magma_free_pinned(ipiv);
  THCTensor_(freeCopyTo)(state, a, ra_);
  THCTensor_(freeCopyTo)(state, b, rb_);
#else
  THError(NoMagma(gesv));
#endif
}

THC_API void THCTensor_(gels)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(b_->nDimension == 2, 1, "b should be 2 dimensional");
  THArgCheck(a_->size[0] == b_->size[0], 2, "size incompatible A,b");
  THArgCheck(a_->size[0] >= a_->size[1], 2, "A should have m >= n");

  THCTensor *a = THCTensor_(newColumnMajor)(state, ra_, a_);
  THCTensor *b = THCTensor_(newColumnMajor)(state, rb_, b_);
  real *a_data = THCTensor_(data)(state, a);
  real *b_data = THCTensor_(data)(state, b);

  int m = a->size[0];
  int n = a->size[1];
  int nrhs = b->size[1];
  real wkopt;

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_sgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);
#else
  magma_dgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);
#endif

  real *hwork = th_magma_malloc_pinned<real>((size_t)wkopt);

#if defined(THC_REAL_IS_FLOAT)
  magma_sgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, hwork, (int)wkopt, &info);
#else
  magma_dgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, hwork, (int)wkopt, &info);
#endif

  magma_free_pinned(hwork);

  if (info != 0)
    THError("MAGMA gels : Argument %d : illegal value", -info);

  THCTensor_(freeCopyTo)(state, a, ra_);
  THCTensor_(freeCopyTo)(state, b, rb_);
#else
  THError(NoMagma(gels));
#endif
}

THC_API void THCTensor_(syev)(THCState *state, THCTensor *re_, THCTensor *rv_, THCTensor *a, const char *jobzs, const char *uplos)
{
#ifdef USE_MAGMA
  int n = a->size[0];
  int lda = n;

  magma_uplo_t uplo = uplos[0] == 'U' ?  MagmaUpper : MagmaLower;
  magma_vec_t jobz = jobzs[0] == 'N' ? MagmaNoVec : MagmaVec;

  THCTensor *input = THCTensor_(newColumnMajor)(state, rv_, a);
  real *input_data = THCTensor_(data)(state, input);

  // eigen values and workspace
  real *w = th_magma_malloc_pinned<real>(n);
  real *wA = th_magma_malloc_pinned<real>(lda);

  // compute optimal size of work array
  int info;
  real lwork;
  int liwork;

#if defined(THC_REAL_IS_FLOAT)
  magma_ssyevd_gpu(jobz, uplo, n, input_data, lda, w, wA, n, &lwork, -1, &liwork, -1, &info);
#else
  magma_dsyevd_gpu(jobz, uplo, n, input_data, lda, w, wA, n, &lwork, -1, &liwork, -1, &info);
#endif

  real *work = th_magma_malloc_pinned<real>((size_t)lwork);
  int *iwork = th_magma_malloc_pinned<int>(liwork);

  // compute eigenvalues and, optionally, eigenvectors
#if defined(THC_REAL_IS_FLOAT)
  magma_ssyevd_gpu(jobz, uplo, n, input_data, lda, w, wA, n, work, (int) lwork, iwork, liwork, &info);
#else
  magma_dsyevd_gpu(jobz, uplo, n, input_data, lda, w, wA, n, work, (int) lwork, iwork, liwork, &info);
#endif

  // copy eigen values from w to re_
  if (info == 0)
    THCTensor_(copyArray1d)(state, re_, w, n);

  magma_free_pinned(iwork);
  magma_free_pinned(work);
  magma_free_pinned(wA);
  magma_free_pinned(w);

  // check error value
  if (info > 0)
    THError("MAGMA syev : Failed to converge. %d off-diagonal elements of an didn't converge to zero", info);
  else if (info < 0)
    THError("MAGMA syev : Argument %d : illegal value", -info);

  THCTensor_(freeCopyTo)(state, input, rv_);
#else
  THError(NoMagma(syev));
#endif
}


#endif

#endif
