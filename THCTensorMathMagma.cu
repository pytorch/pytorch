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
