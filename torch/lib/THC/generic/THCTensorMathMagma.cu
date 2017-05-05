#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMathMagma.cu"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

#ifdef USE_MAGMA

static void THCTensor_(copyArray1d)(THCState *state, THCTensor *self, real *src, int k)
{
  long size[1] = { k };
  long stride[1] = { 1 };
  THCTensor_(resizeNd)(state, self, 1, size, stride);
  size_t len = k * sizeof(real);
  THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THCTensor_(copyArray2d)(THCState *state, THCTensor *self, real *src, int m, int n)
{
  long size[2] = { m, n };
  long stride[2] = { 1, m };
  THCTensor_(resizeNd)(state, self, 2, size, stride);
  size_t len = m * n * sizeof(real);
  THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THCTensor_(copyTensor2d)(THCState *state, real *dst, THCTensor *self)
{
  THAssert(self->nDimension == 2);
  size_t len = THCTensor_(nElement)(state, self)*sizeof(real);
  THCTensor *temp = THCTensor_(newTranspose)(state, self, 0, 1);
  THCTensor *selfc = THCTensor_(newContiguous)(state, temp);
  THCudaCheck(cudaMemcpy(dst, selfc->storage->data + selfc->storageOffset, len, cudaMemcpyDeviceToHost));
  THCTensor_(free)(state, temp);
  THCTensor_(free)(state, selfc);
}

#endif // USE_MAGMA

static THCTensor* THCTensor_(newColumnMajor)(THCState *state, THCTensor *self, THCTensor *src)
{
  THAssert(src->nDimension == 2);
  if (self == src && self->stride[0] == 1 && self->stride[1] == self->size[0])
  {
    THCTensor_(retain)(state, self);
    return self;
  }

  if (self == src)
    self = THCTensor_(new)(state);
  else
    THCTensor_(retain)(state, self);

  long size[2] = { src->size[0], src->size[1] };
  long stride[2] = { 1, src->size[0] };

  THCTensor_(resizeNd)(state, self, 2, size, stride);
  THCTensor_(copy)(state, self, src);
  return self;
}


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

  magma_vec_t jobz = jobus[0] == 'A' ? MagmaAllVec : jobus[0] == 'S' ? MagmaSomeVec : jobus[0] == 'O' ? MagmaOverwriteVec : MagmaNoVec;

  int iunused[1];
  int m = a->size[0];
  int n = a->size[1];
  int k = m < n ? m : n;
  int j = (jobz == MagmaAllVec) ? m : k;
  int jv = (jobz == MagmaAllVec) ? n : k;

  real *a_data = th_magma_malloc_pinned<real>(m * n);
  THCTensor_(copyTensor2d)(state, a_data, a);

  real *rs_data = th_magma_malloc_pinned<real>(k);
  real *ru_data = th_magma_malloc_pinned<real>(m * j);
  real *rv_data = th_magma_malloc_pinned<real>(n * n);

  real wkopt;
  int info;

#if defined(THC_REAL_IS_FLOAT)
  magma_sgesdd(jobz, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, &wkopt, -1, iunused, &info);
#else
  magma_dgesdd(jobz, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, &wkopt, -1, iunused, &info);
#endif

  int lwork = (int) wkopt;
  real *work_data = th_magma_malloc_pinned<real>(lwork);
  int *iwork = th_magma_malloc_pinned<int>(8 * k);

#if defined(THC_REAL_IS_FLOAT)
  magma_sgesdd(jobz, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, work_data, lwork, iwork, &info);
#else
  magma_dgesdd(jobz, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, work_data, lwork, iwork, &info);
#endif

  if (info > 0)
    THError("MAGMA gesdd : the updating process of SBDSDC did not converge (error: %d)", info);
  else if (info < 0)
    THError("MAGMA gesdd : Argument %d : illegal value", -info);

  THCTensor_(copyArray2d)(state, rv_, rv_data, n, n);
  THCTensor_(transpose)(state, rv_, NULL, 0, 1);
  if (jobz != MagmaAllVec)
    THCTensor_(narrow)(state, rv_, rv_, 1, 0, jv);
  THCTensor_(copyArray2d)(state, ru_, ru_data, m, j);
  THCTensor_(copyArray1d)(state, rs_, rs_data, k);
  THCTensor_(copyArray2d)(state, ra_, a_data,  m, n);

  magma_free_pinned(work_data);
  magma_free_pinned(iwork);
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

  THCudaCheck(THCudaFree(state, d_matrices1));
  THCudaCheck(THCudaFree(state, d_matrices1_const));
  THCudaCheck(THCudaFree(state, d_matrices2));

  THCTensor_(freeCopyTo)(state, output, input);
#endif
}

__global__ void THCTensor_(copyUpperSymmetric)(real *input, int n, int len)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
    const int r = idx % n;
    const int c = idx / n;
    if (r > c) {
      input[idx] = input[r*n + c];
    }
  }
}

__global__ void THCTensor_(copyLowerSymmetric)(real *input, int n, int len)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
    const int r = idx % n;
    const int c = idx / n;
    if (r < c) {
      input[idx] = input[r*n + c];
    }
  }
}

THC_API void THCTensor_(potri)(THCState *state, THCTensor *ra_, THCTensor *a, const char *uplo)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];
  magma_uplo_t ul = uplo[0] == 'U' ?  MagmaUpper : MagmaLower;

  THCTensor *input = THCTensor_(newColumnMajor)(state, ra_, a);
  real *input_data = THCTensor_(data)(state, input);

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_spotri_gpu(ul, n, input_data, n, &info);
#else
  magma_dpotri_gpu(ul, n, input_data, n, &info);
#endif

  if (info > 0)
    THError("MAGMA potri : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potri : Argument %d : illegal value", -info);

  cudaStream_t stream = THCState_getCurrentStream(state);
  const int len = n*n;
  dim3 blocks(std::min(DIVUP(len, 128), 65535));
  dim3 threads(128);
  if (uplo[0] == 'U') {
    THCTensor_(copyUpperSymmetric)<<<blocks, threads, 0, stream>>>(input_data, n, len);
  } else {
    THCTensor_(copyLowerSymmetric)<<<blocks, threads, 0, stream>>>(input_data, n, len);
  }

  THCTensor_(freeCopyTo)(state, input, ra_);
#else
  THError(NoMagma(potri));
#endif
}

THC_API void THCTensor_(potrf)(THCState *state, THCTensor *ra_, THCTensor *a, const char *uplo)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];
  magma_uplo_t ul = uplo[0] == 'U' ?  MagmaUpper : MagmaLower;

  THCTensor *input = THCTensor_(newColumnMajor)(state, ra_, a);
  real *input_data = THCTensor_(data)(state, input);

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_spotrf_gpu(ul, n, input_data, n, &info);
#else
  magma_dpotrf_gpu(ul, n, input_data, n, &info);
#endif

  // check error value
  if (info > 0)
    THError("MAGMA potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potrf : Argument %d : illegal value", -info);

  if (uplo[0] == 'U') {
    THCTensor_(triu)(state, ra_, input, 0);
  } else {
    THCTensor_(tril)(state, ra_, input, 0);
  }
  THCTensor_(free)(state, input);
#else
  THError(NoMagma(potrf));
#endif
}

THC_API void THCTensor_(potrs)(THCState *state, THCTensor *rb_, THCTensor *b, THCTensor *a, const char *uplo)
{
#ifdef USE_MAGMA
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];
  int nrhs = b->size[1];
  magma_uplo_t ul = uplo[0] == 'U' ?  MagmaUpper : MagmaLower;

  THCTensor *b_ = THCTensor_(newColumnMajor)(state, rb_, b);
  real *b_data = THCTensor_(data)(state, b_);
  THCTensor *a_ = THCTensor_(newColumnMajor)(state, a, a);
  real *a_data = THCTensor_(data)(state, a_);

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_spotrs_gpu(ul, n, nrhs, a_data, n, b_data, n, &info);
#else
  magma_dpotrs_gpu(ul, n, nrhs, a_data, n, b_data, n, &info);
#endif

  // check error value
  if (info < 0)
    THError("MAGMA potrs : Argument %d : illegal value", -info);

  THCTensor_(freeCopyTo)(state, b_, rb_);
  THCTensor_(free)(state, a_);
#else
  THError(NoMagma(potrs));
#endif
}

THC_API void THCTensor_(qr)(THCState *state, THCTensor *rq_, THCTensor *rr_, THCTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");

  THCTensor *a = THCTensor_(newColumnMajor)(state, rr_, a_);
  int m = a->size[0];
  int n = a->size[1];
  int k = (m < n ? m : n);

#ifdef MAGMA_V2
#if defined(THC_REAL_IS_FLOAT)
  int nb = magma_get_sgeqrf_nb(m, n);
#else
  int nb = magma_get_dgeqrf_nb(m, n);
#endif
#else
#if defined(THC_REAL_IS_FLOAT)
  int nb = magma_get_sgeqrf_nb(m);
#else
  int nb = magma_get_dgeqrf_nb(m);
#endif
#endif

  real *a_data = THCTensor_(data)(state, a);
  real *tau_data = th_magma_malloc_pinned<real>(k);
  THCTensor *work = THCTensor_(newWithSize1d)(state, (2*k + magma_roundup(n, 32))*nb);
  real *work_data = THCTensor_(data)(state, work);

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_sgeqrf2_gpu(m, n, a_data, m, tau_data, &info);
#else
  magma_dgeqrf2_gpu(m, n, a_data, m, tau_data, &info);
#endif

  if (info != 0)
    THError("MAGMA geqrf2 : Argument %d : illegal value.", -info);

  THCTensor_(narrow)(state, a, a, 0, 0, k);
  THCTensor_(triu)(state, rr_, a, 0);
  THCTensor_(free)(state, a);

  a = THCTensor_(newColumnMajor)(state, rq_, a_);
  a_data = THCTensor_(data)(state, a);

#if defined(THC_REAL_IS_FLOAT)
  magma_sgeqrf_gpu(m, n, a_data, m, tau_data, work_data, &info);
#else
  magma_dgeqrf_gpu(m, n, a_data, m, tau_data, work_data, &info);
#endif

  if (info != 0)
    THError("MAGMA geqrf : Argument %d : illegal value.", -info);

  THCTensor *q = THCTensor_(newColumnMajor)(state, rq_, a);
  real *q_data = THCTensor_(data)(state, q);

#if defined(THC_REAL_IS_FLOAT)
  magma_sorgqr_gpu(m, k, k, q_data, m, tau_data, work_data, nb, &info);
#else
  magma_dorgqr_gpu(m, k, k, q_data, m, tau_data, work_data, nb, &info);
#endif

  if (info != 0)
    THError("MAGMA orgqr : Argument %d : illegal value.", -info);

  THCTensor_(free)(state, work);
  magma_free_pinned(tau_data);

  THCTensor_(narrow)(state, q, q, 1, 0, k);
  THCTensor_(freeCopyTo)(state, q, rq_);
#else
  THError(NoMagma(qr));
#endif
}

#endif

#endif
