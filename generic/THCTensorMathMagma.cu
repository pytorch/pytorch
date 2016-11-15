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

THC_API void THCTensor_(geev)(THCState *state, THCTensor *re_, THCTensor *rv_, THCTensor *a_, const char *jobvrs)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 3, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 3, "A should be square");

  magma_vec_t jobvr = jobvrs[0] == 'N' ? MagmaNoVec : MagmaVec;
  int n = a_->size[0];

  real *a_data = th_magma_malloc_pinned<real>(n * n);
  THCTensor_(copyTensor2d)(state, a_data, a_);

  real *wr = th_magma_malloc_pinned<real>(n);
  real *wi = th_magma_malloc_pinned<real>(n);

  real *vr_data = NULL;
  int ldvr = 1;
  if (jobvr == MagmaVec)
  {
    vr_data = th_magma_malloc_pinned<real>(n * n);
    ldvr = n;
  }

  real wkopt;
  int info;

#if defined(THC_REAL_IS_FLOAT)
  magma_sgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, &wkopt, -1, &info);
#else
  magma_dgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, &wkopt, -1, &info);
#endif

  int lwork = (int) wkopt;
  real *work_data = th_magma_malloc_pinned<real>(lwork);

#if defined(THC_REAL_IS_FLOAT)
  magma_sgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, work_data, lwork, &info);
#else
  magma_dgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, work_data, lwork, &info);
#endif

  if (info > 0)
    THError("MAGMA geev : Failed to converge. %d off-diagonal elements of an didn't converge to zero", info);
  else if (info < 0)
    THError("MAGMA geev : Argument %d : illegal value", -info);

  {
    THCTensor_(resize2d)(state, re_, 2, n);
    THCTensor *re = THCTensor_(newContiguous)(state, re_);
    THCudaCheck(cudaMemcpy(re->storage->data + re->storageOffset, wr, n*sizeof(real), cudaMemcpyHostToDevice));
    THCudaCheck(cudaMemcpy(re->storage->data + re->storageOffset + n, wi, n*sizeof(real), cudaMemcpyHostToDevice));
    THCTensor_(freeCopyTo)(state, re, re_);
    THCTensor_(transpose)(state, re_, NULL, 0, 1);
  }

  if (jobvr == MagmaVec)
    THCTensor_(copyArray2d)(state, rv_, vr_data, n, n);

  magma_free_pinned(work_data);
  magma_free_pinned(vr_data);
  magma_free_pinned(wi);
  magma_free_pinned(wr);
  magma_free_pinned(a_data);

#else
  THError(NoMagma(geev));
#endif
}

THC_API void THCTensor_(gesvd)(THCState *state, THCTensor *ru_, THCTensor *rs_, THCTensor *rv_, THCTensor *a, const char *jobu)
{
#ifdef USE_MAGMA
  THCTensor *ra_ = THCTensor_(new)(state);
  THCTensor_(gesvd2)(state, ru_, rs_, rv_,  ra_, a, jobu);
  THCTensor_(free)(state, ra_);
#else
  THError(NoMagma(gesvd));
#endif
}

THC_API void THCTensor_(gesvd2)(THCState *state, THCTensor *ru_, THCTensor *rs_, THCTensor *rv_, THCTensor *ra_, THCTensor *a, const char *jobus)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");

  magma_vec_t jobu = jobus[0] == 'A' ? MagmaAllVec : jobus[0] == 'S' ? MagmaSomeVec : jobus[0] == 'O' ? MagmaOverwriteVec : MagmaNoVec;
  magma_vec_t jobvt = jobu;

  int m = a->size[0];
  int n = a->size[1];
  int k = m < n ? m : n;
  int j = (jobu == MagmaAllVec) ? m : k;

  real *a_data = th_magma_malloc_pinned<real>(m * n);
  THCTensor_(copyTensor2d)(state, a_data, a);

  real *rs_data = th_magma_malloc_pinned<real>(k);
  real *ru_data = th_magma_malloc_pinned<real>(m * j);
  real *rv_data = th_magma_malloc_pinned<real>(n * n);

  real wkopt;
  int info;

#if defined(THC_REAL_IS_FLOAT)
  magma_sgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, &wkopt, -1, &info);
#else
  magma_dgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, &wkopt, -1, &info);
#endif

  int lwork = (int) wkopt;
  real *work_data = th_magma_malloc_pinned<real>(lwork);

#if defined(THC_REAL_IS_FLOAT)
  magma_sgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, work_data, lwork, &info);
#else
  magma_dgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, work_data, lwork, &info);
#endif

  if (info > 0)
    THError("MAGMA gesvd : %d superdiagonals failed to converge", info);
  else if (info < 0)
    THError("MAGMA gesvd : Argument %d : illegal value", -info);

  THCTensor_(copyArray2d)(state, rv_, rv_data, n, n);
  THCTensor_(transpose)(state, rv_, NULL, 0, 1);
  THCTensor_(copyArray2d)(state, ru_, ru_data, m, j);
  THCTensor_(copyArray1d)(state, rs_, rs_data, k);
  THCTensor_(copyArray2d)(state, ra_, a_data,  m, n);

  magma_free_pinned(work_data);
  magma_free_pinned(rv_data);
  magma_free_pinned(ru_data);
  magma_free_pinned(rs_data);
  magma_free_pinned(a_data);
#else
  THError(NoMagma(gesvd2));
#endif
}

THC_API void THCTensor_(getri)(THCState *state, THCTensor *ra_, THCTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int info;
  int n = a->size[0];
  int lwork = n * magma_get_sgetri_nb(n);

  THCTensor *input = THCTensor_(newColumnMajor)(state, ra_, a);
  real *input_data = THCTensor_(data)(state, input);

  int *ipiv = th_magma_malloc_pinned<int>(n);

  THCTensor *work = THCTensor_(newWithSize1d)(state, lwork);
  real *work_data = THCTensor_(data)(state, work);

  // Run LU
#if defined(THC_REAL_IS_FLOAT)
  magma_sgetrf_gpu(n, n, input_data, n, ipiv, &info);
#else
  magma_dgetrf_gpu(n, n, input_data, n, ipiv, &info);
#endif

  if (info > 0)
    THError("MAGMA getrf : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("MAGMA getrf : Argument %d : illegal value", -info);

  // Inverse
#if defined(THC_REAL_IS_FLOAT)
  magma_sgetri_gpu(n, input_data, n, ipiv, work_data, lwork, &info);
#else
  magma_dgetri_gpu(n, input_data, n, ipiv, work_data, lwork, &info);
#endif

  if (info > 0)
    THError("MAGMA getri : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("MAGMA getri : Argument %d : illegal value", -info);

  THCTensor_(free)(state, work);
  magma_free_pinned(ipiv);
  THCTensor_(freeCopyTo)(state, input, ra_);
#else
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];

  // input
  THCTensor *input = THCTensor_(newColumnMajor)(state, ra_, a);
  // output
  THCTensor *output = THCTensor_(newColumnMajor)(state, ra_, a);

  size_t matrices_size = sizeof(real*);

  real **matrices1 = (real **)THAlloc(matrices_size);
  const real **matrices1_const = (const real **)THAlloc(matrices_size);
  real **matrices2 = (real **)THAlloc(matrices_size);
  matrices1[0] = THCTensor_(data)(state, input);
  matrices1_const[0] = THCTensor_(data)(state, input);
  matrices2[0] = THCTensor_(data)(state, output);

  // Copy pointers to device.
  real **d_matrices1, **d_matrices2;
  const real **d_matrices1_const;
  THCudaCheck(THCudaMalloc(state, (void**)&d_matrices1, matrices_size));
  THCudaCheck(THCudaMalloc(state, (void**)&d_matrices1_const, matrices_size));
  THCudaCheck(THCudaMalloc(state, (void**)&d_matrices2, matrices_size));

  THCudaCheck(cudaMemcpyAsync(d_matrices1, matrices1, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));
  THCudaCheck(cudaMemcpyAsync(d_matrices1_const, matrices1_const, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));
  THCudaCheck(cudaMemcpyAsync(d_matrices2, matrices2, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));
  int info;
  int *info_gpu;
  THCudaCheck(THCudaMalloc(state, (void**)&info_gpu, sizeof(int)));

  int *ipiv_gpu;
  THCudaCheck(THCudaMalloc(state, (void**)&ipiv_gpu, n * sizeof(int)));

  // Run LU
#if defined(THC_REAL_IS_FLOAT)
  THCudaBlas_Sgetrf(state, n, d_matrices1, n, ipiv_gpu, info_gpu, 1);
#else
  THCudaBlas_Dgetrf(state, n, d_matrices1, n, ipiv_gpu, info_gpu, 1);
#endif

  THCudaCheck(cudaMemcpy(&info, info_gpu, sizeof(int), cudaMemcpyDeviceToHost));

  if (info > 0)
    THError("CUBLAS getrf : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("CUBLAS getrf : Argument %d : illegal value", -info);

  // Inverse
#if defined(THC_REAL_IS_FLOAT)
  THCudaBlas_Sgetri(state, n, d_matrices1_const, n, ipiv_gpu, d_matrices2, n, info_gpu, 1);
#else
  THCudaBlas_Dgetri(state, n, d_matrices1_const, n, ipiv_gpu, d_matrices2, n, info_gpu, 1);
#endif

  if (info > 0)
    THError("CUBLAS getri : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("CUBLAS getri : Argument %d : illegal value", -info);

  THCudaCheck(THCudaFree(state, ipiv_gpu));
  THCudaCheck(THCudaFree(state, info_gpu));
  THCTensor_(freeCopyTo)(state, output, input);
#endif
}

#endif

#endif
