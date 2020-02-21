#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathMagma.cu"
#else

#if defined(THC_REAL_IS_FLOAT) || defined(THC_REAL_IS_DOUBLE)

#ifdef USE_MAGMA

static void THCTensor_(copyArray1d)(THCState *state, THCTensor *self, scalar_t *src, int k)
{
  int64_t size[1] = { k };
  int64_t stride[1] = { 1 };
  THCTensor_(resizeNd)(state, self, 1, size, stride);
  size_t len = k * sizeof(scalar_t);
  THCudaCheck(cudaMemcpy(THCStorage_(data)(state, THTensor_getStoragePtr(self)) + self->storage_offset(), src, len, cudaMemcpyHostToDevice));
}

static void THCTensor_(copyArray2d)(THCState *state, THCTensor *self, scalar_t *src, int m, int n)
{
  int64_t size[2] = { m, n };
  int64_t stride[2] = { 1, m };
  THCTensor_(resizeNd)(state, self, 2, size, stride);
  size_t len = m * n * sizeof(scalar_t);
  THCudaCheck(cudaMemcpy(THCStorage_(data)(state, THTensor_getStoragePtr(self)) + self->storage_offset(), src, len, cudaMemcpyHostToDevice));
}

static void THCTensor_(copyTensor2d)(THCState *state, scalar_t *dst, THCTensor *self)
{
  THAssert(self->dim() == 2);
  size_t len = THCTensor_(nElement)(state, self)*sizeof(scalar_t);
  THCTensor *temp = THCTensor_(newTranspose)(state, self, 0, 1);
  THCTensor *selfc = THCTensor_(newContiguous)(state, temp);
  THCudaCheck(cudaMemcpy(dst, THCStorage_(data)(state, THTensor_getStoragePtr(selfc)) + selfc->storage_offset(), len, cudaMemcpyDeviceToHost));
  THCTensor_(free)(state, temp);
  THCTensor_(free)(state, selfc);
}

#endif // USE_MAGMA

static THCTensor* THCTensor_(newColumnMajor)(THCState *state, THCTensor *self, THCTensor *src)
{
  THAssert(src->dim() == 2);
  if (self == src && self->stride(0) == 1 && self->stride(1) == self->size(0))
  {
    THCTensor_(retain)(state, self);
    return self;
  }

  if (self == src)
    self = THCTensor_(new)(state);
  else
    THCTensor_(retain)(state, self);

  int64_t size[2] = { src->size(0), src->size(1) };
  int64_t stride[2] = { 1, src->size(0) };

  THCTensor_(resizeNd)(state, self, 2, size, stride);
  THCTensor_(copy)(state, self, src);
  return self;
}

void THCTensor_(gels)(THCState *state, THCTensor *rb_, THCTensor *ra_, THCTensor *b_, THCTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(!a_->is_empty() && a_->dim() == 2, 1, "A should be (non-empty) 2 dimensional");
  THArgCheck(!b_->is_empty() && b_->dim() == 2, 1, "b should be (non-empty) 2 dimensional");
  TORCH_CHECK(a_->size(0) == b_->size(0), "Expected A and b to have same size "
      "at dim 0, but A has ", a_->size(0), " rows and B has ", b_->size(0), " rows");
  THArgCheck(a_->size(0) >= a_->size(1), 2, "Expected A with shape (m x n) to have "
      "m >= n. The case for m < n is not implemented yet.");

  THCTensor *a = THCTensor_(newColumnMajor)(state, ra_, a_);
  THCTensor *b = THCTensor_(newColumnMajor)(state, rb_, b_);
  scalar_t *a_data = THCTensor_(data)(state, a);
  scalar_t *b_data = THCTensor_(data)(state, b);

  int64_t m = a->size(0);
  int64_t n = a->size(1);
  int64_t nrhs = b->size(1);
  scalar_t wkopt;

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_sgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);
#else
  magma_dgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);
#endif

  scalar_t *hwork = th_magma_malloc_pinned<scalar_t>((size_t)wkopt);

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

void THCTensor_(geev)(THCState *state, THCTensor *re_, THCTensor *rv_, THCTensor *a_, bool eigenvectors)
{
#ifdef USE_MAGMA
  char jobvrs = eigenvectors ? 'V' : 'N';
  THArgCheck(a_->dim() == 2, 3, "A should be 2 dimensional");
  THArgCheck(a_->size(0) == a_->size(1), 3, "A should be square");

  magma_vec_t jobvr = jobvrs == 'N' ? MagmaNoVec : MagmaVec;
  int64_t n = a_->size(0);

  scalar_t *a_data = th_magma_malloc_pinned<scalar_t>(n * n);
  THCTensor_(copyTensor2d)(state, a_data, a_);

  scalar_t *wr = th_magma_malloc_pinned<scalar_t>(n);
  scalar_t *wi = th_magma_malloc_pinned<scalar_t>(n);

  scalar_t *vr_data = NULL;
  int64_t ldvr = 1;
  if (jobvr == MagmaVec)
  {
    vr_data = th_magma_malloc_pinned<scalar_t>(n * n);
    ldvr = n;
  }

  scalar_t *work_data = nullptr;

  if (n > 0) {
    int info;
    scalar_t wkopt;
#if defined(THC_REAL_IS_FLOAT)
    magma_sgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, &wkopt, -1, &info);
#else
    magma_dgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, &wkopt, -1, &info);
#endif

    int lwork = (int) wkopt;
    work_data = th_magma_malloc_pinned<scalar_t>(lwork);

#if defined(THC_REAL_IS_FLOAT)
    magma_sgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, work_data, lwork, &info);
#else
    magma_dgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, work_data, lwork, &info);
#endif

    if (info > 0)
      THError("MAGMA geev : Failed to converge. %d off-diagonal elements of an didn't converge to zero", info);
    else if (info < 0)
      THError("MAGMA geev : Argument %d : illegal value", -info);
  }

  {
    THCTensor_(resize2d)(state, re_, 2, n);
    THCTensor *re = THCTensor_(newContiguous)(state, re_);
    if (n > 0) {
      THCudaCheck(cudaMemcpy(THCStorage_(data)(state, THTensor_getStoragePtr(re)) + re->storage_offset(), wr, n*sizeof(scalar_t), cudaMemcpyHostToDevice));
      THCudaCheck(cudaMemcpy(THCStorage_(data)(state, THTensor_getStoragePtr(re)) + re->storage_offset() + n, wi, n*sizeof(scalar_t), cudaMemcpyHostToDevice));
    }
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

__global__ void THCTensor_(copyUpperSymmetric)(scalar_t *input, int n, int len)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
    const int r = idx % n;
    const int c = idx / n;
    if (r > c) {
      input[idx] = input[r*n + c];
    }
  }
}

__global__ void THCTensor_(copyLowerSymmetric)(scalar_t *input, int n, int len)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
    const int r = idx % n;
    const int c = idx / n;
    if (r < c) {
      input[idx] = input[r*n + c];
    }
  }
}

void THCTensor_(potri)(THCState *state, THCTensor *ra_, THCTensor *a, bool upper)
{
  char uplo = upper ? 'U' : 'L';
#ifdef USE_MAGMA
  THArgCheck(!a->is_empty() && a->dim() == 2, 2, "A should be non-empty 2 dimensional");
  THArgCheck(a->size(0) == a->size(1), 2, "A should be square");

  int64_t n = a->size(0);
  magma_uplo_t ul = uplo == 'U' ?  MagmaUpper : MagmaLower;

  THCTensor *input = THCTensor_(newColumnMajor)(state, ra_, a);
  scalar_t *input_data = THCTensor_(data)(state, input);

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

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const int len = n*n;
  dim3 blocks(std::min(DIVUP(len, 128), 65535));
  dim3 threads(128);
  if (uplo == 'U') {
    THCTensor_(copyUpperSymmetric)<<<blocks, threads, 0, stream>>>(input_data, n, len);
  } else {
    THCTensor_(copyLowerSymmetric)<<<blocks, threads, 0, stream>>>(input_data, n, len);
  }

  THCTensor_(freeCopyTo)(state, input, ra_);
#else
  THError(NoMagma(potri));
#endif
}

void THCTensor_(geqrf)(THCState *state, THCTensor *ra_, THCTensor *rtau_, THCTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(!a_->is_empty() && a_->dim() == 2, 2, "A should be non-empty 2 dimensional");

  THCTensor *a = THCTensor_(newColumnMajor)(state, ra_, a_);
  int64_t m = a->size(0);
  int64_t n = a->size(1);
  int64_t k = (m < n ? m : n);

#if defined(THC_REAL_IS_FLOAT)
  int64_t nb = magma_get_sgeqrf_nb(m, n);
#else
  int64_t nb = magma_get_dgeqrf_nb(m, n);
#endif

  scalar_t *rtau_data = th_magma_malloc_pinned<scalar_t>(k);
  scalar_t *a_data = THCTensor_(data)(state, a);

  int info;
#if defined(THC_REAL_IS_FLOAT)
  magma_sgeqrf2_gpu(m, n, a_data, m, rtau_data, &info);
#else
  magma_dgeqrf2_gpu(m, n, a_data, m, rtau_data, &info);
#endif

  if (info != 0)
    THError("MAGMA geqrf2 : Argument %d : illegal value.", -info);

  THCTensor_(freeCopyTo)(state, a, ra_);
  THCTensor_(copyArray1d)(state, rtau_, rtau_data, k);
  magma_free_pinned(rtau_data);
#else
  THError(NoMagma(geqrf));
#endif
}

#endif

#endif
