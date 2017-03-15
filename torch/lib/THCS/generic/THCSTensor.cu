#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.cu"
#else

#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#define I_INFO(tensor) getTensorInfo<THCIndexTensor, unsigned long>(state, tensor)
#define V_INFO(tensor) getTensorInfo<THCTensor, unsigned long>(state, tensor)

THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self) {
  THLongStorage *storage;
  THCTensor *other;

  THCSTensor_(contiguous)(state, self);

  // set up the new tensor
  storage = THCSTensor_(newSizeOf)(state, self);
  other = THCTensor_(newWithSize)(state, storage, NULL);
  THCTensor_(zero)(state, other);

  const ptrdiff_t nnz = THCSTensor_(nnz)(state, self);
  if (nnz == 0) {
    THLongStorage_free(storage);
    return other;
  }

  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, nnz, grid), 1, CUTORCH_DIM_WARNING);

  THCSTensor_spcKernel<TensorAddOp<real>, unsigned long, real>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        TensorAddOp<real>(),
        V_INFO(other), I_INFO(self->indices), V_INFO(self->values),
        (unsigned long)(nnz));

  THCudaCheck(cudaGetLastError());
  THLongStorage_free(storage);
  return other;
}

void THCSTensor_(reorder)(THCState *state, THCSTensor *self) {
  if (self->nnz < 2) return;
#if CUDA_VERSION >= 7000
  THCThrustAllocator thrustAlloc(state);
#define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
#else
#define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
#endif

  THCIndexTensor *indices = THCSTensor_(indices)(state, self);
  THCIndexTensor *indicesSlice = THCIndexTensor_(new)(state);
  THCudaLongTensor *permutation = THCudaLongTensor_newWithSize1d(state, self->nnz);

  // Sort indices in lexicographic order (following thrust's example recipe)
  thrust::device_ptr<long> permutationIter(THCudaLongTensor_data(state, permutation));
  thrust::device_vector<integer> indicesBuffer(self->nnz);
  THRUST_EXEC(thrust::sequence, permutationIter, permutationIter + self->nnz);

  for (int i = self->nDimensionI - 1; i >= 0; i--) {
    THCIndexTensor_(select)(state, indicesSlice, indices, 0, i);
    thrust::device_ptr<integer> indicesIter(THCIndexTensor_(data)(state, indicesSlice));
    THRUST_EXEC(thrust::gather, permutationIter, permutationIter + self->nnz, indicesIter, indicesBuffer.begin());
    THRUST_EXEC(thrust::stable_sort_by_key, indicesBuffer.begin(), indicesBuffer.end(), permutationIter);
  }

  for (int i = 0; i < self->nDimensionI; i++) {
    THCIndexTensor_(select)(state, indicesSlice, indices, 0, i);
    thrust::device_ptr<integer> indicesIter(THCIndexTensor_(data)(state, indicesSlice));
    THRUST_EXEC(thrust::copy, indicesIter, indicesIter + self->nnz, indicesBuffer.begin());
    THRUST_EXEC(thrust::gather, permutationIter, permutationIter + self->nnz, indicesBuffer.begin(), indicesIter);
  }

  THCTensor *values = THCSTensor_(values)(state, self);
  THCTensor *newValues = THCTensor_(new)(state);
  THCTensor_(indexSelect)(state, newValues, values, 0, permutation);
  THCTensor_(free)(state, values);
  THCTensor_(free)(state, self->values);
  self->values = newValues;

  // Make indices unique
  // TODO for the moment the only parallelism is when copying/adding non-scalar values
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, newValues->stride[0], grid), 1, CUTORCH_DIM_WARNING);

  THCSTensor_uniqueValuesReorderKernel<unsigned long, real>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        I_INFO(indices), V_INFO(newValues), (unsigned long)(self->nnz));
  THCudaCheck(cudaGetLastError());

  THCudaLongStorage *resultNnz = THCudaLongStorage_newWithSize(state, 1);
  THCSTensor_uniqueIndicesReorderKernel<unsigned long, real>
    <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        I_INFO(indices), (unsigned long)(self->nnz), (unsigned long*)resultNnz->data);
  THCudaCheck(cudaGetLastError());
  self->nnz = THCudaLongStorage_get(state, resultNnz, 0);
  THCudaLongStorage_free(state, resultNnz);

  THCudaLongTensor_free(state, permutation);
  THCIndexTensor_(free)(state, indicesSlice);
  THCIndexTensor_(free)(state, indices);

#undef THRUST_EXEC
}

void THCSTensor_(contiguous)(THCState *state, THCSTensor *self) {
  if (self->contiguous) return;
  THCSTensor_(reorder)(state, self);
  self->contiguous = 1;
}

// In place transpose
void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int d1, int d2) {
  /* TODO
  THCudaLongTensor *indices = THCSTensor_(indices)(state, self);
  long i;
  for (i = 0; i < THCSTensor_(nnz)(state, self); i++) {
    long tmp = THCTensor_fastGet2d(indices, d1, i);
    THCTensor_fastSet2d(indices, d1, i,
        THCTensor_fastGet2d(indices, d2, i));
    THCTensor_fastSet2d(indices, d2, i, tmp);
  }
  i = self->size[d1];
  self->size[d1] = self->size[d2];
  self->size[d2] = i;
  self->contiguous = 0;
  THFree(indices);
  */
  THError("WARNING: Sparse Cuda Tensor op transpose is not implemented");
}

int THCSTensor_(getDevice)(THCState* state, const THCSTensor* tensor) {
  if (!tensor->values || !tensor->values->storage) return -1;
  return THCStorage_(getDevice)(state, tensor->values->storage);
}

#endif
