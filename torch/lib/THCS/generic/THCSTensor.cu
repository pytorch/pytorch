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
  THLongStorage_free(storage);
  THCTensor_(zero)(state, other);

  const ptrdiff_t nnz = THCSTensor_(nnz)(state, self);
  if (nnz == 0) {
    return other;
  }

  // TODO more benchmarking
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (self->nDimensionV == 0) {
    THArgCheck(getApplyGrid(state, nnz, grid), 1, CUTORCH_DIM_WARNING);

    THCSTensor_sparseElementwiseKernelScalar<TensorAddOp<real>, unsigned long, real>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          TensorAddOp<real>(),
          V_INFO(other), I_INFO(self->indices), V_INFO(self->values),
          (unsigned long)(nnz));
  } else {
    THArgCheck(getApplyGrid(state, nnz * block.x, grid), 1, CUTORCH_DIM_WARNING);

    THCSTensor_sparseElementwiseKernel<TensorAddOp<real>, unsigned long, real>
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(
          TensorAddOp<real>(),
          V_INFO(other), I_INFO(self->indices), V_INFO(self->values),
          (unsigned long)(nnz));
  }

  THCudaCheck(cudaGetLastError());
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

  // For indices, a simple sort + unique suffices
  // For values, we reduce the problem to a sparse x dense matrix multiplication D2 = S x D1, such that:
  // * D1 represents the input values, D2 represents the output values
  // * D1 and D2 are views over the values where:
  //    * the first dimension represents the nnz index (same as in the values tensor)
  //    * the second dimension represents the "flattened" values (so they can be treated as blocks of scalars even if they are N-dimensional)
  // * S maps values in D1 to their position in D2
  //   Multiple values in D1 can map to the same position in D2 if there are duplicate indices
  //   Values mapping to the same position are added together (which is what matrix multiplication does)
  //
  // When constructing S, we must make sure that it is contiguous (otherwise this function will call itself when doing the multiplication)
  // To achieve this, we define the indices tensor of S as follows:
  // * the second row contains the permutation corresponding to a stable sort of the original indices
  // * the first row "maps" those indices to their final location after deduplication
  //
  // The construction of the second row ensures that the first row is sorted
  // Because the sorting used for the second row is stable, groups of values mapped to the same position correspond to increasing subsequences of the permutation
  // So the indices tensor of S is guaranteed to be sorted

  // Initialize tensors
  THCIndexTensor *indices = THCSTensor_(indices)(state, self);
  THCTensor *values = THCSTensor_(values)(state, self);
  THCudaLongTensor *sIndices = THCudaLongTensor_newWithSize2d(state, 2, self->nnz);
  THCTensor *sValues = THCTensor_(newWithSize1d)(state, self->nnz);
  THCTensor_(fill)(state, sValues, ScalarConvert<int, real>::to(1));
  THCudaLongTensor *mapping = THCudaLongTensor_new(state);
  THCudaLongTensor *permutation = THCudaLongTensor_new(state);
  THCudaLongTensor_select(state, mapping, sIndices, 0, 0);
  THCudaLongTensor_select(state, permutation, sIndices, 0, 1);
  THCudaLongTensor *uniquePositions = THCudaLongTensor_newWithSize1d(state, self->nnz);

  // convert N-dimensional indices to scalar indices
  THCIndexTensor *indicesScalar = THCIndexTensor_(newWithSize1d)(state, self->nnz);
  THCIndexTensor *indicesSlice = THCIndexTensor_(new)(state);
  THCIndexTensor_(zero)(state, indicesScalar);
  integer factor = 1;
  for (int i = self->nDimensionI - 1; i >= 0; i--) {
    THCIndexTensor_(select)(state, indicesSlice, indices, 0, i);
    THCIndexTensor_(cadd)(state, indicesScalar, indicesScalar, factor, indicesSlice);
    factor *= self->size[i];
  }
  THCIndexTensor_(free)(state, indicesSlice);

  // stable sort indices and remember the permutation
  thrust::device_ptr<long> permutationIter(THCudaLongTensor_data(state, permutation));
  THRUST_EXEC(thrust::sequence, permutationIter, permutationIter + self->nnz);
  thrust::device_ptr<integer> indicesIter(THCIndexTensor_(data)(state, indicesScalar));
  THRUST_EXEC(thrust::stable_sort_by_key, indicesIter, indicesIter + self->nnz, permutationIter);
  // Note: the code below is much faster and seems to work, but the sort is not stable.
  // It could be that csrmm2 works even when column indices are not sorted within rows
  // THCIndexTensor *indicesScalarClone = THCIndexTensor_(newClone)(state, indicesScalar);
  // THCIndexTensor_(sort)(state, indicesScalar, permutation, indicesScalarClone, 0, 0);
  // THCIndexTensor_(free)(state, indicesScalarClone);

  // compute a list of unique indices, along with their position in the original index tensor (using the saved permutation)
  thrust::device_ptr<long> uniquePositionsIter(THCudaLongTensor_data(state, uniquePositions));
  thrust::device_vector<integer> uniqueIndicesBuffer(self->nnz); // not used, can we optimize?
  thrust::pair<thrust::device_vector<integer>::iterator, thrust::device_ptr<long> > newEnd =
    THRUST_EXEC(thrust::unique_by_key_copy, indicesIter, indicesIter + self->nnz, permutationIter, uniqueIndicesBuffer.begin(), uniquePositionsIter);
  long newNnz = newEnd.second - uniquePositionsIter;
  THCudaLongTensor_resize1d(state, uniquePositions, newNnz);

  // compute the mapping of sorted indices to their final location after deduplication
  THCudaLongTensor_set1d(state, mapping, 0, 0);
  thrust::device_ptr<long> mappingIter(THCudaLongTensor_data(state, mapping));
  thrust::not_equal_to<integer> op;
  THRUST_EXEC(thrust::transform, indicesIter, indicesIter + self->nnz - 1, indicesIter + 1, mappingIter + 1, op);
  THRUST_EXEC(thrust::inclusive_scan, mappingIter, mappingIter + self->nnz, mappingIter);

  // build S
  THCSTensor *S = THCSTensor_(newWithSize2d)(state, newNnz, self->nnz);
  THCSTensor_(_move)(state, S, sIndices, sValues);
  S->contiguous = 1;

  // build output indices tensor by doing an indexSelect over the sorted list of unique indices
  THCIndexTensor *newIndices = THCIndexTensor_(new)(state);
  THCIndexTensor_(indexSelect)(state, newIndices, indices, 1, uniquePositions);
  THCIndexTensor_(free)(state, indices);
  THCIndexTensor_(free)(state, self->indices);
  self->indices = newIndices;

  // create output values tensor
  THLongStorage *newValuesSizes = THCTensor_(newSizeOf)(state, values);
  THLongStorage_set(newValuesSizes, 0, newNnz);
  THCTensor *newValues = THCTensor_(newWithSize)(state, newValuesSizes, NULL);
  THLongStorage_free(newValuesSizes);

  // create D1: view over input values tensor
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

  // create D2: view over output values tensor
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

  // build output values tensor by computing D2 = S x D1
  THCSTensor_(spaddmm)(state, newValuesView, ScalarConvert<int, real>::to(0), newValuesView, ScalarConvert<int, real>::to(1), S, valuesView);
  THCTensor_(free)(state, values);
  THCTensor_(free)(state, self->values);
  self->values = newValues;

  self->nnz = newNnz;

  THCudaLongTensor_free(state, permutation);
  THCudaLongTensor_free(state, mapping);
  THCudaLongTensor_free(state, uniquePositions);
  THCIndexTensor_(free)(state, indicesScalar);
  THCSTensor_(free)(state, S);
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
  long nDimI = THCSTensor_(nDimensionI)(state, self);
  long nDimV = THCSTensor_(nDimensionV)(state, self);
  THArgCheck(d1 < nDimI && d2 < nDimI, 1, "Transposed dimensions should be sparse. Got nDimI: %ld, d1: %ld, d2: %ld", nDimI, d1, d2);
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
