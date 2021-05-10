#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathPointwise.cu"
#else

#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>

#if !defined(THC_REAL_IS_BOOL)

static void propagate_names_if_named_tensor_enabled(THCTensor* result, THCTensor* src) {
  at::namedinference::propagate_names(result, src);
}

void THCTensor_(crossKernel)(THCState *state, THCTensor *self, THCTensor *x, THCTensor *y, int dimension)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self, x, y));

  int64_t sx = THCTensor_(stride)(state, x, dimension);
  int64_t sy = THCTensor_(stride)(state, y, dimension);
  int64_t so = THCTensor_(stride)(state, self, dimension);
  THCTensor *nx = THCTensor_(newNarrow)(state, x, dimension, 0, 1);
  THCTensor *ny = THCTensor_(newNarrow)(state, y, dimension, 0, 1);
  THCTensor *nself = THCTensor_(newNarrow)(state, self, dimension, 0, 1);
  if (!THC_pointwiseApply3<scalar_t, scalar_t, scalar_t>(state, nself, nx, ny, TensorCrossOp<scalar_t>(sx, sy, so))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }
  THCTensor_(free)(state, nx);
  THCTensor_(free)(state, ny);
  THCTensor_(free)(state, nself);
}
#endif
#endif
