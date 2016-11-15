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
