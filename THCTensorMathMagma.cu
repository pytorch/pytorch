#include "THCGeneral.h"
#include "THCTensorMath.h"
#include "THCTensorCopy.h"
#include <algorithm>

#ifdef USE_MAGMA
#include <magma.h>
#else
#include "THCBlas.h"
#endif

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define NoMagma(name) "No CUDA implementation of '" #name "'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/)"

void THCMagma_init(THCState *state)
{
#ifdef USE_MAGMA
  magma_init();
#endif
}

#ifdef USE_MAGMA
static inline float* th_magma_smalloc_pinned(size_t n)
{
  float* ptr;
  if (MAGMA_SUCCESS != magma_smalloc_pinned(&ptr, n))
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", n/268435456);
  return ptr;
}

static inline int* th_magma_imalloc_pinned(size_t n)
{
  int* ptr;
  if (MAGMA_SUCCESS != magma_imalloc_pinned(&ptr, n))
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", n/268435456);
  return ptr;
}

static void THCudaTensor_copyArray1d(THCState *state, THCudaTensor *self, float *src, int k)
{
  long size[1] = { k };
  long stride[1] = { 1 };
  THCudaTensor_rawResize(state, self, 1, size, stride);
  size_t len = k * sizeof(float);
  THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THCudaTensor_copyArray2d(THCState *state, THCudaTensor *self, float *src, int m, int n)
{
  long size[2] = { m, n };
  long stride[2] = { 1, m };
  THCudaTensor_rawResize(state, self, 2, size, stride);
  size_t len = m * n * sizeof(float);
  THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THCudaTensor_copyTensor2d(THCState *state, float *dst, THCudaTensor *self)
{
  THAssert(self->nDimension == 2);
  size_t len = THCudaTensor_nElement(state, self)*sizeof(float);
  THCudaTensor *temp = THCudaTensor_newTranspose(state, self, 0, 1);
  THCudaTensor *selfc = THCudaTensor_newContiguous(state, temp);
  THCudaCheck(cudaMemcpy(dst, selfc->storage->data + selfc->storageOffset, len, cudaMemcpyDeviceToHost));
  THCudaTensor_free(state, temp);
  THCudaTensor_free(state, selfc);
}

#endif

static THCudaTensor* THCudaTensor_newColumnMajor(THCState *state, THCudaTensor *self, THCudaTensor *src)
{
  THAssert(src->nDimension == 2);
  if (self == src && self->stride[0] == 1 && self->stride[1] == self->size[0])
  {
    THCudaTensor_retain(state, self);
    return self;
  }

  if (self == src)
    self = THCudaTensor_new(state);
  else
    THCudaTensor_retain(state, self);

  long size[2] = { src->size[0], src->size[1] };
  long stride[2] = { 1, src->size[0] };

  THCudaTensor_rawResize(state, self, 2, size, stride);
  THCudaTensor_copy(state, self, src);
  return self;
}


void THCudaTensor_gesv(THCState *state, THCudaTensor *rb_, THCudaTensor *ra_, THCudaTensor *b_, THCudaTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(b_->nDimension == 2, 2, "b should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 1, "A should be square");
  THArgCheck(b_->size[0] == a_->size[0], 2, "A,b size incompatible");

  int n = a_->size[0];
  int nrhs = b_->size[1];

  THCudaTensor *a = THCudaTensor_newColumnMajor(state, ra_, a_);
  THCudaTensor *b = THCudaTensor_newColumnMajor(state, rb_, b_);
  float *a_data = THCudaTensor_data(state, a);
  float *b_data = THCudaTensor_data(state, b);

  int *ipiv = th_magma_imalloc_pinned(n);

  int info;
  magma_sgesv_gpu(n, nrhs, a_data, n, ipiv, b_data, n, &info);

  if (info < 0)
    THError("MAGMA gesv : Argument %d : illegal value", -info);
  else if (info > 0)
    THError("MAGMA gesv : U(%d,%d) is zero, singular U.", info, info);

  magma_free_pinned(ipiv);
  THCudaTensor_freeCopyTo(state, a, ra_);
  THCudaTensor_freeCopyTo(state, b, rb_);
#else
  THError(NoMagma(gesv));
#endif
}

void THCudaTensor_gels(THCState *state, THCudaTensor *rb_, THCudaTensor *ra_, THCudaTensor *b_, THCudaTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(b_->nDimension == 2, 1, "b should be 2 dimensional");
  THArgCheck(a_->size[0] == b_->size[0], 2, "size incompatible A,b");
  THArgCheck(a_->size[0] >= a_->size[1], 2, "A should have m >= n");

  THCudaTensor *a = THCudaTensor_newColumnMajor(state, ra_, a_);
  THCudaTensor *b = THCudaTensor_newColumnMajor(state, rb_, b_);
  float *a_data = THCudaTensor_data(state, a);
  float *b_data = THCudaTensor_data(state, b);

  int m = a->size[0];
  int n = a->size[1];
  int nrhs = b->size[1];
  float wkopt;

  int info;
  magma_sgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);

  float *hwork = th_magma_smalloc_pinned((size_t)wkopt);
  magma_sgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, hwork, (int)wkopt, &info);
  magma_free_pinned(hwork);

  if (info != 0)
    THError("MAGMA gels : Argument %d : illegal value", -info);

  THCudaTensor_freeCopyTo(state, a, ra_);
  THCudaTensor_freeCopyTo(state, b, rb_);
#else
  THError(NoMagma(gels));
#endif
}

void THCudaTensor_syev(THCState *state, THCudaTensor *re_, THCudaTensor *rv_, THCudaTensor *a, const char *jobzs, const char *uplos)
{
#ifdef USE_MAGMA
  int n = a->size[0];
  int lda = n;

  magma_uplo_t uplo = uplos[0] == 'U' ?  MagmaUpper : MagmaLower;
  magma_vec_t jobz = jobzs[0] == 'N' ? MagmaNoVec : MagmaVec;

  THCudaTensor *input = THCudaTensor_newColumnMajor(state, rv_, a);
  float *input_data = THCudaTensor_data(state, input);

  // eigen values and workspace
  float *w = th_magma_smalloc_pinned(n);
  float *wA = th_magma_smalloc_pinned(lda);

  // compute optimal size of work array
  int info;
  float lwork;
  int liwork;
  magma_ssyevd_gpu(jobz, uplo, n, input_data, lda, w, wA, n, &lwork, -1, &liwork, -1, &info);

  float *work = th_magma_smalloc_pinned((size_t)lwork);
  int *iwork = th_magma_imalloc_pinned(liwork);

  // compute eigenvalues and, optionally, eigenvectors
  magma_ssyevd_gpu(jobz, uplo, n, input_data, lda, w, wA, n, work, (int) lwork, iwork, liwork, &info);

  // copy eigen values from w to re_
  if (info == 0)
    THCudaTensor_copyArray1d(state, re_, w, n);

  magma_free_pinned(iwork);
  magma_free_pinned(work);
  magma_free_pinned(wA);
  magma_free_pinned(w);

  // check error value
  if (info > 0)
    THError("MAGMA syev : Failed to converge. %d off-diagonal elements of an didn't converge to zero", info);
  else if (info < 0)
    THError("MAGMA syev : Argument %d : illegal value", -info);

  THCudaTensor_freeCopyTo(state, input, rv_);
#else
  THError(NoMagma(syev));
#endif
}

void THCudaTensor_geev(THCState *state, THCudaTensor *re_, THCudaTensor *rv_, THCudaTensor *a_, const char *jobvrs)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 3, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 3, "A should be square");

  magma_vec_t jobvr = jobvrs[0] == 'N' ? MagmaNoVec : MagmaVec;
  int n = a_->size[0];

  float *a_data = th_magma_smalloc_pinned(n * n);
  THCudaTensor_copyTensor2d(state, a_data, a_);

  float *wr = th_magma_smalloc_pinned(n);
  float *wi = th_magma_smalloc_pinned(n);

  float *vr_data = NULL;
  int ldvr = 1;
  if (jobvr == MagmaVec)
  {
    vr_data = th_magma_smalloc_pinned(n * n);
    ldvr = n;
  }

  float wkopt;
  int info;

  magma_sgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, &wkopt, -1, &info);

  int lwork = (int) wkopt;
  float *work_data = th_magma_smalloc_pinned(lwork);

  magma_sgeev(MagmaNoVec, jobvr, n, a_data, n, wr, wi, NULL, 1, vr_data, ldvr, work_data, lwork, &info);

  if (info > 0)
    THError("MAGMA geev : Failed to converge. %d off-diagonal elements of an didn't converge to zero", info);
  else if (info < 0)
    THError("MAGMA geev : Argument %d : illegal value", -info);

  {
    THCudaTensor_resize2d(state, re_, 2, n);
    THCudaTensor *re = THCudaTensor_newContiguous(state, re_);
    THCudaCheck(cudaMemcpy(re->storage->data + re->storageOffset, wr, n*sizeof(float), cudaMemcpyHostToDevice));
    THCudaCheck(cudaMemcpy(re->storage->data + re->storageOffset + n, wi, n*sizeof(float), cudaMemcpyHostToDevice));
    THCudaTensor_freeCopyTo(state, re, re_);
    THCudaTensor_transpose(state, re_, NULL, 0, 1);
  }

  if (jobvr == MagmaVec)
    THCudaTensor_copyArray2d(state, rv_, vr_data, n, n);

  magma_free_pinned(work_data);
  magma_free_pinned(vr_data);
  magma_free_pinned(wi);
  magma_free_pinned(wr);
  magma_free_pinned(a_data);

#else
  THError(NoMagma(geev));
#endif
}

void THCudaTensor_gesvd(THCState *state, THCudaTensor *ru_, THCudaTensor *rs_, THCudaTensor *rv_, THCudaTensor *a, const char *jobu)
{
#ifdef USE_MAGMA
  THCudaTensor *ra_ = THCudaTensor_new(state);
  THCudaTensor_gesvd2(state, ru_, rs_, rv_,  ra_, a, jobu);
  THCudaTensor_free(state, ra_);
#else
  THError(NoMagma(gesvd));
#endif
}

void THCudaTensor_gesvd2(THCState *state, THCudaTensor *ru_, THCudaTensor *rs_, THCudaTensor *rv_, THCudaTensor *ra_, THCudaTensor *a, const char *jobus)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");

  magma_vec_t jobu = jobus[0] == 'A' ? MagmaAllVec : jobus[0] == 'S' ? MagmaSomeVec : jobus[0] == 'O' ? MagmaOverwriteVec : MagmaNoVec;
  magma_vec_t jobvt = jobu;

  int m = a->size[0];
  int n = a->size[1];
  int k = m < n ? m : n;
  int j = (jobu == MagmaAllVec) ? m : k;

  float *a_data = th_magma_smalloc_pinned(m * n);
  THCudaTensor_copyTensor2d(state, a_data, a);

  float *rs_data = th_magma_smalloc_pinned(k);
  float *ru_data = th_magma_smalloc_pinned(m * j);
  float *rv_data = th_magma_smalloc_pinned(n * n);

  float wkopt;
  int info;
  magma_sgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, &wkopt, -1, &info);

  int lwork = (int) wkopt;
  float *work_data = th_magma_smalloc_pinned(lwork);

  magma_sgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, work_data, lwork, &info);

  if (info > 0)
    THError("MAGMA gesvd : %d superdiagonals failed to converge", info);
  else if (info < 0)
    THError("MAGMA gesvd : Argument %d : illegal value", -info);

  THCudaTensor_copyArray2d(state, rv_, rv_data, n, n);
  THCudaTensor_transpose(state, rv_, NULL, 0, 1);
  THCudaTensor_copyArray2d(state, ru_, ru_data, m, j);
  THCudaTensor_copyArray1d(state, rs_, rs_data, k);
  THCudaTensor_copyArray2d(state, ra_, a_data,  m, n);

  magma_free_pinned(work_data);
  magma_free_pinned(rv_data);
  magma_free_pinned(ru_data);
  magma_free_pinned(rs_data);
  magma_free_pinned(a_data);
#else
  THError(NoMagma(gesvd2));
#endif
}

void THCudaTensor_getri(THCState *state, THCudaTensor *ra_, THCudaTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int info;
  int n = a->size[0];
  int lwork = n * magma_get_sgetri_nb(n);

  THCudaTensor *input = THCudaTensor_newColumnMajor(state, ra_, a);
  float *input_data = THCudaTensor_data(state, input);

  int *ipiv = th_magma_imalloc_pinned(n);

  THCudaTensor *work = THCudaTensor_newWithSize1d(state, lwork);
  float *work_data = THCudaTensor_data(state, work);

  // Run LU
  magma_sgetrf_gpu(n, n, input_data, n, ipiv, &info);
  if (info > 0)
    THError("MAGMA getrf : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("MAGMA getrf : Argument %d : illegal value", -info);

  // Inverse
  magma_sgetri_gpu(n, input_data, n, ipiv, work_data, lwork, &info);
  if (info > 0)
    THError("MAGMA getri : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("MAGMA getri : Argument %d : illegal value", -info);

  THCudaTensor_free(state, work);
  magma_free_pinned(ipiv);
  THCudaTensor_freeCopyTo(state, input, ra_);
#else
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];

  // input
  THCudaTensor *input = THCudaTensor_newColumnMajor(state, ra_, a);
  // output
  THCudaTensor *output = THCudaTensor_newColumnMajor(state, ra_, a);

  size_t matrices_size = sizeof(float*);

  float **matrices1 = (float **)THAlloc(matrices_size);
  const float **matrices1_const = (const float **)THAlloc(matrices_size);
  float **matrices2 = (float **)THAlloc(matrices_size);
  matrices1[0] = THCudaTensor_data(state, input);
  matrices1_const[0] = THCudaTensor_data(state, input);
  matrices2[0] = THCudaTensor_data(state, output);

  // Copy pointers to device.
  float **d_matrices1, **d_matrices2;
  const float **d_matrices1_const;
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
  THCudaBlas_getrf(state, n, d_matrices1, n, ipiv_gpu, info_gpu, 1);

  THCudaCheck(cudaMemcpy(&info, info_gpu, sizeof(int), cudaMemcpyDeviceToHost));

  if (info > 0)
    THError("CUBLAS getrf : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("CUBLAS getrf : Argument %d : illegal value", -info);

  // Inverse
  THCudaBlas_getri(state, n, d_matrices1_const, n, ipiv_gpu, d_matrices2, n, info_gpu, 1);
  if (info > 0)
    THError("CUBLAS getri : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("CUBLAS getri : Argument %d : illegal value", -info);

  THCudaCheck(THCudaFree(state, ipiv_gpu));
  THCudaCheck(THCudaFree(state, info_gpu));
  THCudaTensor_freeCopyTo(state, output, input);
#endif

}


__global__ void THCudaTensor_copyUpperSymmetric(float *input, int n, int len)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
    const int r = idx % n;
    const int c = idx / n;
    if (r > c) {
      input[idx] = input[r*n + c];
    }
  }
}

void THCudaTensor_potri(THCState *state, THCudaTensor *ra_, THCudaTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];

  THCudaTensor *input = THCudaTensor_newColumnMajor(state, ra_, a);
  float *input_data = THCudaTensor_data(state, input);

  int info;
  magma_spotrf_gpu(MagmaUpper, n, input_data, n, &info);
  if (info > 0)
    THError("MAGMA potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potrf : Argument %d : illegal value", -info);

  magma_spotri_gpu(MagmaUpper, n, input_data, n, &info);
  if (info > 0)
    THError("MAGMA potri : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potri : Argument %d : illegal value", -info);

  cudaStream_t stream = THCState_getCurrentStream(state);
  const int len = n*n;
  dim3 blocks(std::min(DIVUP(len, 128), 65535));
  dim3 threads(128);
  THCudaTensor_copyUpperSymmetric<<<blocks, threads, 0, stream>>>(input_data, n, len);

  THCudaTensor_freeCopyTo(state, input, ra_);
#else
  THError(NoMagma(potri));
#endif
}

void THCudaTensor_potrf(THCState *state, THCudaTensor *ra_, THCudaTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];

  THCudaTensor *input = THCudaTensor_newColumnMajor(state, ra_, a);
  float *input_data = THCudaTensor_data(state, input);

  int info;
  magma_spotrf_gpu(MagmaUpper, n, input_data, n, &info);

  // check error value
  if (info > 0)
    THError("MAGMA potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potrf : Argument %d : illegal value", -info);

  THCudaTensor_triu(state, ra_, input, 0);
  THCudaTensor_free(state, input);
#else
  THError(NoMagma(potrf));
#endif
}

void THCudaTensor_potrs(THCState *state, THCudaTensor *rb_, THCudaTensor *b, THCudaTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];
  int nrhs = b->size[1];

  THCudaTensor *b_ = THCudaTensor_newColumnMajor(state, rb_, b);
  float *b_data = THCudaTensor_data(state, b_);
  THCudaTensor *a_ = THCudaTensor_newColumnMajor(state, a, a);
  float *a_data = THCudaTensor_data(state, a_);

  int info;
  magma_spotrs_gpu(MagmaUpper, n, nrhs, a_data, n, b_data, n, &info);

  // check error value
  if (info < 0)
    THError("MAGMA potrs : Argument %d : illegal value", -info);

  THCudaTensor_freeCopyTo(state, b_, rb_);
  THCudaTensor_free(state, a_);
#else
  THError(NoMagma(potrs));
#endif
}

void THCudaTensor_qr(THCState *state, THCudaTensor *rq_, THCudaTensor *rr_, THCudaTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");

  THCudaTensor *a = THCudaTensor_newColumnMajor(state, rr_, a_);
  int m = a->size[0];
  int n = a->size[1];
  int k = (m < n ? m : n);

#ifdef MAGMA_V2
  int nb = magma_get_sgeqrf_nb(m, n);
#else
  int nb = magma_get_sgeqrf_nb(m);
#endif

  float *a_data = THCudaTensor_data(state, a);
  float *tau_data = th_magma_smalloc_pinned(n*n);

  THCudaTensor *work = THCudaTensor_newWithSize1d(state, (2*k + ((n+31)/32)*32)*nb);
  float *work_data = THCudaTensor_data(state, work);

  int info;
  magma_sgeqrf_gpu(m, n, a_data, m, tau_data, work_data, &info);

  if (info != 0)
    THError("MAGMA geqrf : Argument %d : illegal value.", -info);

  THCudaTensor *q = THCudaTensor_newColumnMajor(state, rq_, a);
  float *q_data = THCudaTensor_data(state, q);

  THCudaTensor_narrow(state, a, a, 0, 0, k);
  THCudaTensor_triu(state, rr_, a, 0);
  THCudaTensor_free(state, a);

  magma_sorgqr_gpu(m, n, k, q_data, m, tau_data, work_data, nb, &info);

  if (info != 0)
    THError("MAGMA orgqr : Argument %d : illegal value.", -info);

  THCudaTensor_free(state, work);
  magma_free_pinned(tau_data);

  THCudaTensor_narrow(state, q, q, 1, 0, k);
  THCudaTensor_freeCopyTo(state, q, rq_);
#else
  THError(NoMagma(qr));
#endif
}
