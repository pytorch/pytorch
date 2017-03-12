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

THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self) {
  THLongStorage *storage;
  THCTensor *other;

  THCSTensor_(contiguous)(state, self);

  // set up the new tensor
  storage = THCSTensor_(newSizeOf)(state, self);
  other = THCTensor_(newWithSize)(state, storage, NULL);
  THCTensor_(zero)(state, other);

  // Some necessary dimensions and sizes
  const ptrdiff_t nnz = THCSTensor_(nnz)(state, self);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, nnz, grid), 1, CUTORCH_DIM_WARNING);

  TensorInfo<real, unsigned long> otherInfo =
    getTensorInfo<THCTensor, unsigned long>(state, other);
  TensorInfo<long, unsigned long> indicesInfo =
    getTensorInfo<THCudaLongTensor, unsigned long>(state, self->indices);
  TensorInfo<real, unsigned long> valuesInfo =
    getTensorInfo<THCTensor, unsigned long>(state, self->values);

  THCSTensor_toDenseKernel<unsigned long, real>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        otherInfo, indicesInfo, valuesInfo, (unsigned long)(nnz));

  THCudaCheck(cudaGetLastError());
  THLongStorage_free(storage);
  return other;
}

void THCSTensor_(reorder)(THCState *state, THCSTensor *self) {
#if CUDA_VERSION >= 7000
  THCThrustAllocator thrustAlloc(state);
#define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
#else
#define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
#endif

  THCIndexTensor *indices = THCSTensor_(indices)(state, self);
  THCIndexTensor *indicesSlice = THCIndexTensor_(new)(state);
  THCIndexTensor *permutation = THCIndexTensor_(newWithSize1d)(state, self->nnz);

  // Sort indices in lexicographic order (following thrust's example recipe)
  thrust::device_ptr<integer> permutationIter(THCIndexTensor_(data)(state, permutation));
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

  THCTensor *newValues = THCTensor_(new)(state);
  THCTensor_(indexSelect)(state, newValues, self->values, 0, permutation);
  THCTensor_(free)(state, self->values);
  self->values = newValues;

  // Some necessary dimensions and sizes
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, self->nnz, grid), 1, CUTORCH_DIM_WARNING);

  TensorInfo<integer, unsigned long> indicesInfo =
    getTensorInfo<THCudaLongTensor, unsigned long>(state, self->indices);
  TensorInfo<real, unsigned long> valuesInfo =
    getTensorInfo<THCTensor, unsigned long>(state, self->values);

  THCSTensor_uniqueValuesReorderKernel<unsigned long, real>
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
        indicesInfo, valuesInfo, (unsigned long)(self->nnz));
  THCudaCheck(cudaGetLastError());

  THCudaLongStorage *resultNnz = THCudaLongStorage_newWithSize(state, 1);
  THCSTensor_uniqueIndicesReorderKernel<unsigned long, real>
    <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        indicesInfo, (unsigned long)(self->nnz), (unsigned long*)resultNnz->data);
  THCudaCheck(cudaGetLastError());
  self->nnz = THCudaLongStorage_get(state, resultNnz, 0);
  THCudaLongStorage_free(state, resultNnz);

  THCIndexTensor_(free)(state, permutation);
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

void THCTensor_(sparseMask)(THCState *state, THCSTensor *r_, THCTensor *t, THCSTensor *mask) {
  THError("WARNING: Sparse Cuda Tensor op sparseMask is not implemented");
}

#endif
