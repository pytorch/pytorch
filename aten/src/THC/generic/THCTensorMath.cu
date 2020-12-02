#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMath.cu"
#else

#include <algorithm>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/MemoryOverlap.h>

void THCTensor_(fill)(THCState* state, THCTensor *self_, scalar_t value)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));

  if (!THC_pointwiseApply1<scalar_t>(
        state, self_, TensorFillOp<scalar_t>(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(zero)(THCState *state, THCTensor *self_)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 1, self_));
  if (THCTensor_(isContiguous)(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCTensor_(data)(state, self_),
                                0,
                                sizeof(scalar_t) * THCTensor_(nElement)(state, self_),
                                c10::cuda::getCurrentCUDAStream()));
  } else {
    if (!THC_pointwiseApply1<scalar_t>(
          state, self_,
          TensorFillOp<scalar_t>(ScalarConvert<int, scalar_t>::to(0)))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

ptrdiff_t
THCTensor_(numel)(THCState *state, THCTensor *t)
{
  return THCTensor_(nElement)(state, t);
}

void THCTensor_(check_shape_except_dim)(THCState *state,
    THCTensor *first, THCTensor *second, int dimension, int index);
inline void THCTensor_(check_shape_except_dim)(THCState *state,
    THCTensor *first, THCTensor *second, int dimension, int index)
{
  int first_dims = first->dim();
  int second_dims = second->dim();
  THArgCheck(first_dims == second_dims, 0,
      "Tensors must have same number of dimensions: got %d and %d",
      first_dims, second_dims);
  for (int dim = 0; dim < first_dims; dim++) {
    if (dim == dimension) {
      continue;
    }
    int64_t first_dim_size = THCTensor_(size)(state, first, dim);
    int64_t second_dim_size = THCTensor_(size)(state, second, dim);
    THArgCheck(first_dim_size == second_dim_size, 0,
        "Sizes of tensors must match except in dimension %d. Got %lld and %lld in dimension %d (The offending index is %d)",
        dimension, (long long)first_dim_size, (long long)second_dim_size, dim, index);
  }
}


#endif
