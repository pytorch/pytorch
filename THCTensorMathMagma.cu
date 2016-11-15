#include "THCGeneral.h"
#include "THCTensorMath.h"
#include "THCTensorCopy.h"
#include "THCTensorMathMagma.cuh"
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

  int *ipiv = th_magma_malloc_pinned<int>(n);

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
  THCudaBlas_Sgetrf(state, n, d_matrices1, n, ipiv_gpu, info_gpu, 1);

  THCudaCheck(cudaMemcpy(&info, info_gpu, sizeof(int), cudaMemcpyDeviceToHost));

  if (info > 0)
    THError("CUBLAS getrf : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("CUBLAS getrf : Argument %d : illegal value", -info);

  // Inverse
  THCudaBlas_Sgetri(state, n, d_matrices1_const, n, ipiv_gpu, d_matrices2, n, info_gpu, 1);
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

__global__ void THCudaTensor_copyLowerSymmetric(float *input, int n, int len)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
    const int r = idx % n;
    const int c = idx / n;
    if (r < c) {
      input[idx] = input[r*n + c];
    }
  }
}

void THCudaTensor_potri(THCState *state, THCudaTensor *ra_, THCudaTensor *a, const char *uplo)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];
  magma_uplo_t ul = uplo[0] == 'U' ?  MagmaUpper : MagmaLower;

  THCudaTensor *input = THCudaTensor_newColumnMajor(state, ra_, a);
  float *input_data = THCudaTensor_data(state, input);

  int info;
  magma_spotri_gpu(ul, n, input_data, n, &info);
  if (info > 0)
    THError("MAGMA potri : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potri : Argument %d : illegal value", -info);

  cudaStream_t stream = THCState_getCurrentStream(state);
  const int len = n*n;
  dim3 blocks(std::min(DIVUP(len, 128), 65535));
  dim3 threads(128);
  if (uplo[0] == 'U') {
    THCudaTensor_copyUpperSymmetric<<<blocks, threads, 0, stream>>>(input_data, n, len);
  } else {
    THCudaTensor_copyLowerSymmetric<<<blocks, threads, 0, stream>>>(input_data, n, len);
  }

  THCudaTensor_freeCopyTo(state, input, ra_);
#else
  THError(NoMagma(potri));
#endif
}

void THCudaTensor_potrf(THCState *state, THCudaTensor *ra_, THCudaTensor *a, const char *uplo)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];
  magma_uplo_t ul = uplo[0] == 'U' ?  MagmaUpper : MagmaLower;

  THCudaTensor *input = THCudaTensor_newColumnMajor(state, ra_, a);
  float *input_data = THCudaTensor_data(state, input);

  int info;
  magma_spotrf_gpu(ul, n, input_data, n, &info);

  // check error value
  if (info > 0)
    THError("MAGMA potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potrf : Argument %d : illegal value", -info);

  if (uplo[0] == 'U') {
    THCudaTensor_triu(state, ra_, input, 0);
  } else {
    THCudaTensor_tril(state, ra_, input, 0);
  }
  THCudaTensor_free(state, input);
#else
  THError(NoMagma(potrf));
#endif
}

void THCudaTensor_potrs(THCState *state, THCudaTensor *rb_, THCudaTensor *b, THCudaTensor *a, const char *uplo)
{
#ifdef USE_MAGMA
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];
  int nrhs = b->size[1];
  magma_uplo_t ul = uplo[0] == 'U' ?  MagmaUpper : MagmaLower;

  THCudaTensor *b_ = THCudaTensor_newColumnMajor(state, rb_, b);
  float *b_data = THCudaTensor_data(state, b_);
  THCudaTensor *a_ = THCudaTensor_newColumnMajor(state, a, a);
  float *a_data = THCudaTensor_data(state, a_);

  int info;
  magma_spotrs_gpu(ul, n, nrhs, a_data, n, b_data, n, &info);

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
  float *tau_data = th_magma_malloc_pinned<float>(n*n);

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

#include "generic/THCTensorMathMagma.cu"
#include "THCGenerateAllTypes.h"
