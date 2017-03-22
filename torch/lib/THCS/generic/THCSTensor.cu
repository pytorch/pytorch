#ifndef THCS_GENERIC_FILE
#define THCS_GENERIC_FILE "generic/THCSTensor.cu"
#else

#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include <thrust/generate.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
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

  // TODO more benchmarking
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (self->nDimensionV == 0) {
    THArgCheck(getApplyGrid(state, nnz, grid), 1, CUTORCH_DIM_WARNING);

    THCSTensor_spcKernelScalar<TensorAddOp<real>, unsigned long, real>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          TensorAddOp<real>(),
          V_INFO(other), I_INFO(self->indices), V_INFO(self->values),
          (unsigned long)(nnz));
  } else {
    THArgCheck(getApplyGrid(state, nnz * block.x, grid), 1, CUTORCH_DIM_WARNING);

    THCSTensor_spcKernel<TensorAddOp<real>, unsigned long, real>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          TensorAddOp<real>(),
          V_INFO(other), I_INFO(self->indices), V_INFO(self->values),
          (unsigned long)(nnz));
  }

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
  THCTensor *values = THCSTensor_(values)(state, self);
  THCIndexTensor *indicesSlice = THCIndexTensor_(new)(state);
  THCIndexTensor *indicesScalar = THCIndexTensor_(newWithSize1d)(state, self->nnz);
  THCIndexTensor *projectIndices = THCIndexTensor_(newWithSize2d)(state, 2, self->nnz);
  THCTensor *projectValues = THCTensor_(newWithSize1d)(state, self->nnz);
  THCTensor_(fill)(state, projectValues, ScalarConvert<int, real>::to(1));
  THCudaLongTensor *permutation = THCudaLongTensor_new(state);
  THCudaLongTensor *mapping = THCudaLongTensor_new(state);
  THCudaLongTensor_select(state, mapping, projectIndices, 0, 0);
  THCudaLongTensor_select(state, permutation, projectIndices, 0, 1);
  THCudaLongTensor *unique = THCudaLongTensor_newWithSize1d(state, self->nnz);

  thrust::device_ptr<long> permutationIter(THCudaLongTensor_data(state, permutation));
  THRUST_EXEC(thrust::sequence, permutationIter, permutationIter + self->nnz);
  THCIndexTensor_(zero)(state, indicesScalar);
  integer factor = 1;
  for (int i = self->nDimensionI - 1; i >= 0; i--) {
    THCIndexTensor_(select)(state, indicesSlice, indices, 0, i);
    THCIndexTensor_(cadd)(state, indicesScalar, indicesScalar, factor, indicesSlice);
    factor *= self->size[i];
  }
  thrust::device_ptr<integer> indicesIter(THCIndexTensor_(data)(state, indicesScalar));
  THRUST_EXEC(thrust::stable_sort_by_key, indicesIter, indicesIter + self->nnz, permutationIter);
  thrust::device_ptr<long> uniqueIter(THCudaLongTensor_data(state, unique));
  thrust::device_vector<integer> indicesBuffer(self->nnz); // not used, can we optimize?
  thrust::pair<thrust::device_vector<integer>::iterator, thrust::device_ptr<long> > newEnd =
    THRUST_EXEC(thrust::unique_by_key_copy, indicesIter, indicesIter + self->nnz, permutationIter, indicesBuffer.begin(), uniqueIter);
  long newNnz = newEnd.second - uniqueIter;
  THCudaLongTensor_resize1d(state, unique, newNnz);
  THCudaLongTensor_set1d(state, mapping, 0, 0);
  thrust::device_ptr<long> mappingIter(THCudaLongTensor_data(state, mapping));

  thrust::not_equal_to<integer> op;
  THRUST_EXEC(thrust::transform, indicesIter, indicesIter + self->nnz - 1, indicesIter + 1, mappingIter + 1, op);
  THRUST_EXEC(thrust::inclusive_scan, mappingIter, mappingIter + self->nnz, mappingIter);

  THCIndexTensor *newIndices = THCIndexTensor_(new)(state);
  THCIndexTensor_(indexSelect)(state, newIndices, indices, 1, unique);
  THCIndexTensor_(free)(state, indices);
  THCIndexTensor_(free)(state, self->indices);
  self->indices = newIndices;

  THCSTensor *project = THCSTensor_(newWithSize2d)(state, newNnz, self->nnz);
  THCSTensor_(move)(state, project, projectIndices, projectValues);
  project->contiguous = 1;
  THLongStorage *newValuesSizes = THCTensor_(newSizeOf)(state, values);
  THLongStorage_set(newValuesSizes, 0, newNnz);
  THCTensor *newValues = THCTensor_(newWithSize)(state, newValuesSizes, NULL);
  THLongStorage_free(newValuesSizes);

  THCTensor *valuesView;
  if (THCTensor_(nDimension)(state, values) != 2) {
    THLongStorage *valuesViewSizes = THLongStorage_newWithSize(2);
    THLongStorage_set(valuesViewSizes, 0, self->nnz);
    THLongStorage_set(valuesViewSizes, 1, -1);
    valuesView = THCTensor_(newView)(state, values, valuesViewSizes);
    THLongStorage_free(valuesViewSizes);
  } else {
    valuesView = values;
  }

  THCTensor *newValuesView;
  if (THCTensor_(nDimension)(state, newValues) != 2) {
    THLongStorage *newValuesViewSizes = THLongStorage_newWithSize(2);
    THLongStorage_set(newValuesViewSizes, 0, newNnz);
    THLongStorage_set(newValuesViewSizes, 1, -1);
    newValuesView = THCTensor_(newView)(state, newValues, newValuesViewSizes);
    THLongStorage_free(newValuesViewSizes);
  } else {
    newValuesView = newValues;
  }

  THCSTensor_(spaddmm)(state, newValuesView, ScalarConvert<int, real>::to(0), newValuesView, ScalarConvert<int, real>::to(1), project, valuesView);
  THCTensor_(free)(state, values);
  THCTensor_(free)(state, self->values);
  self->values = newValues;

  self->nnz = newNnz;

  THCudaLongTensor_free(state, permutation);
  THCudaLongTensor_free(state, mapping);
  THCudaLongTensor_free(state, unique);
  THCIndexTensor_(free)(state, indicesSlice);
  THCIndexTensor_(free)(state, indicesScalar);
  THCSTensor_(free)(state, project);
  if (valuesView != values) {
    THCTensor_(free)(state, valuesView);
  }
  if (newValuesView != newValues) {
    THCTensor_(free)(state, newValuesView);
  }

#undef THRUST_EXEC
}

// In place transpose
void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int d1, int d2) {
  THCIndexTensor *indices = THCSTensor_(indices)(state, self);
  long nnz = THCSTensor_(nnz)(state, self);
  THCIndexTensor *buffer = THCIndexTensor_(newWithSize1d)(state, nnz);
  THCIndexTensor *slice1 = THCIndexTensor_(new)(state);
  THCIndexTensor *slice2 = THCIndexTensor_(new)(state);
  THCIndexTensor_(select)(state, slice1, indices, 0, d1);
  THCIndexTensor_(select)(state, slice2, indices, 0, d2);
  THCIndexTensor_(copy)(state, buffer, slice1);
  THCIndexTensor_(copy)(state, slice1, slice2);
  THCIndexTensor_(copy)(state, slice2, buffer);
  long i = self->size[d1];
  self->size[d1] = self->size[d2];
  self->size[d2] = i;
  self->contiguous = 0;
  THCIndexTensor_(free)(state, indices);
  THCIndexTensor_(free)(state, buffer);
  THCIndexTensor_(free)(state, slice1);
  THCIndexTensor_(free)(state, slice2);
}

int THCSTensor_(getDevice)(THCState* state, const THCSTensor* tensor) {
  if (!tensor->values || !tensor->values->storage) return -1;
  return THCStorage_(getDevice)(state, tensor->values->storage);
}

#endif
