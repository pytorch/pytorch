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

#define I_INFO(tensor) getTensorInfo<THCIndexTensor, uint64_t>(state, tensor)
#define V_INFO(tensor) getTensorInfo<THCTensor, uint64_t>(state, tensor)

THCTensor *THCSTensor_(toDense)(THCState *state, THCSTensor *self) {
  THLongStorage *size;
  THCTensor *dst;

  // set up the new tensor
  size = THCSTensor_(newSizeOf)(state, self);
  dst = THCTensor_(newWithSize)(state, size, NULL);
  THLongStorage_free(size);
  THCTensor_(zero)(state, dst);

  real one = ScalarConvert<int, real>::to(1);
  THCSTensor_(spcadd)(state, dst, dst, one, self);
  THCudaCheck(cudaGetLastError());
  return dst;
}

THCSTensor *THCSTensor_(newCoalesce)(THCState *state, THCSTensor *self) {
  ptrdiff_t nnz = self->nnz;
  if (nnz < 2) {
    self->coalesced = 1;
  }
  if (self->coalesced) {
    THCSTensor_(retain)(state, self);
    return self;
  }

#if CUDA_VERSION >= 7000
  THCThrustAllocator thrustAlloc(state);
#define THRUST_EXEC(fn, ...) fn(thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)), ##__VA_ARGS__)
#else
#define THRUST_EXEC(fn, ...) fn(##__VA_ARGS__)
#endif

  // For indices, a simple sort + unique suffices
  // For values, we use a custom kernel for segmented reduction (can't use Thrust due to indirection).

  THCTensor *values_ = THCSTensor_(newValues)(state, self);
  THCTensor *values = THCTensor_(newContiguous)(state, values_);
  THCTensor_(free)(state, values_);

  int nDimI = self->nDimensionI;
  int64_t stride = values->stride[0];

  cudaStream_t stream = THCState_getCurrentStream(state);

  // indices will be modified by Thrust, so we have to clone or use new storage
  // here.
  THCIndexTensor *indices1D = THCSTensor_(newFlattenedIndices)(state, self, 1);

  THCIndexTensor *origIndices = THCIndexTensor_(newWithSize1d)(state, nnz);
  THCIndexTensor *uniqueOffsets = THCIndexTensor_(newWithSize1d)(state, nnz);

  typedef thrust::device_ptr<int64_t> thrust_ptr;
  thrust_ptr indicesIter(THCIndexTensor_(data)(state, indices1D));
  thrust_ptr origIndicesIter(THCIndexTensor_(data)(state, origIndices));
  thrust_ptr uniqueOffsetsIter(THCIndexTensor_(data)(state, uniqueOffsets));


  // Fill sortedOrigIndices with sequential indices
  thrust::counting_iterator<int64_t> countIterI(TH_INDEX_BASE);
  thrust::counting_iterator<int64_t> countIterO(TH_INDEX_BASE);

  THRUST_EXEC(thrust::copy, countIterI, countIterI + nnz, origIndicesIter);
  THRUST_EXEC(thrust::copy, countIterO, countIterO + nnz, uniqueOffsetsIter);

  THRUST_EXEC(thrust::sort_by_key,
    indicesIter, indicesIter + nnz,
    origIndicesIter, ThrustLTOp<int64_t>()
  );

  // this forces device-host synchronization!
  thrust::pair<thrust_ptr, thrust_ptr> newEnd = THRUST_EXEC(
    thrust::unique_by_key,
    indicesIter, indicesIter + nnz,
    uniqueOffsetsIter
  );
  int64_t newNnz = newEnd.first - indicesIter;

  THCIndexTensor_(resize2d)(state, indices1D, 1, newNnz);
  THCTensor *newValues = THCTensor_(new)(state);
  THCTensor_(resizeNd)(state, newValues, values->nDimension, values->size, NULL);
  newValues->size[0] = newNnz;


  dim3 grid(THCCeilDiv(newNnz, (int64_t) 4), THCCeilDiv(stride, (int64_t) 128));
  dim3 block(32, 4);
  THCSTensor_coalesceValuesKernel<real, accreal><<<grid, block, 0, stream>>>(
    THCIndexTensor_(data)(state, uniqueOffsets),
    THCIndexTensor_(data)(state, origIndices),
    THCTensor_(data)(state, values),
    THCTensor_(data)(state, newValues),
    nnz,
    newNnz,
    stride
  );

// this grid-strided version is slower but probably more flexible
  // to different sizes
  // int64_t blockX = min(stride, (int64_t) 512);
  // dim3 block(blockX, 512 / blockX);
  // int64_t grid = min((int64_t) 1024, THCCeilDiv((int64_t) newNnz * stride, (int64_t) block.x * block.y));
  // THCSTensor_coalesceValuesKernel_gridStrided<real, accreal><<<grid, block, 0, stream>>>(
  //   THCIndexTensor_(data)(state, uniqueOffsets),
  //   THCIndexTensor_(data)(state, origIndices),
  //   THCTensor_(data)(state, values),
  //   THCTensor_(data)(state, newValues),
  //   nnz,
  //   newNnz,
  //   stride
  // );

  THCIndexTensor_(free)(state, origIndices);
  THCIndexTensor_(free)(state, uniqueOffsets);

  ////////////////////////////////////////////////////////////
  // unflatten indices if necessary
  THCIndexTensor *newIndices;
  if (nDimI == 1) {
    newIndices = indices1D;
  } else {
    newIndices = THCIndexTensor_(newWithSize2d)(state, nDimI, newNnz);
    THCIndexTensor *indicesSlice = THCIndexTensor_(new)(state);
    if (TH_INDEX_BASE != 0) {
      THCIndexTensor_(add)(state, indices1D, indices1D, -1);
    }
    for (int64_t d = nDimI - 1; d >= 0; d--) {
      THCIndexTensor_(select)(state, indicesSlice, newIndices, 0, d);
      THCIndexTensor_(copy)(state, indicesSlice, indices1D);
      THCIndexTensor_(div)(state, indices1D, indices1D, self->size[d]);
      THCIndexTensor_(cadd)(state, indicesSlice, indicesSlice, -self->size[d], indices1D);
    }
    if (TH_INDEX_BASE != 0) {
      THCIndexTensor_(add)(state, newIndices, newIndices, 1);
    }
    THCIndexTensor_(free)(state, indices1D);
    THCIndexTensor_(free)(state, indicesSlice);
  }
  ////////////////////////////////////////////////////////////
  THLongStorage *size = THCSTensor_(newSizeOf)(state, self);
  THCSTensor *dst = THCSTensor_(newWithTensorAndSize)(state, newIndices, newValues, size);
  THLongStorage_free(size);

  THCTensor_(free)(state, values);
  THCIndexTensor_(free)(state, newIndices);
  THCTensor_(free)(state, newValues);

  dst->coalesced = 1;
  THCudaCheck(cudaGetLastError());
  return dst;
#undef THRUST_EXEC
}

// forceClone is intended to use as a boolean, if set, the result will forced to
// be a clone of self.
THCIndexTensor* THCSTensor_(newFlattenedIndices)(THCState *state, THCSTensor *self, int forceClone) {
  THCIndexTensor *indices = THCSTensor_(newIndices)(state, self);
  int nDimI = self->nDimensionI;
  if (nDimI == 1) {
    if (forceClone) {
      THCIndexTensor *indices_clone = THCIndexTensor_(newClone)(state, indices);
      THCIndexTensor_(free)(state, indices);
      return indices_clone;
    } else {
      return indices;
    }
  } else {
    // FIXME TH_INDEX_BASE
    int64_t factor = 1;
    THCIndexTensor *indices1D = THCIndexTensor_(newWithSize2d)(state, 1, self->nnz);
    THCIndexTensor_(fill)(state, indices1D, TH_INDEX_BASE);
    THCIndexTensor *indicesSlice = THCIndexTensor_(new)(state);
    for (int64_t d = nDimI - 1; d >= 0; d--) {
      THCIndexTensor_(select)(state, indicesSlice, indices, 0, d);
      THCIndexTensor_(cadd)(state, indices1D, indices1D, factor, indicesSlice);
      if (TH_INDEX_BASE != 0) {
        THCIndexTensor_(add)(state, indices1D, indices1D, -TH_INDEX_BASE);
      }
      factor *= self->size[d];
    }
    THCIndexTensor_(free)(state, indices);
    THCIndexTensor_(free)(state, indicesSlice);
    return indices1D;
  }
}

// In place transpose
void THCSTensor_(transpose)(THCState *state, THCSTensor *self, int d1, int d2) {
  int64_t nDimI = THCSTensor_(nDimensionI)(state, self);
  int64_t nDimV = THCSTensor_(nDimensionV)(state, self);
  THArgCheck(d1 < nDimI && d2 < nDimI, 1, "Transposed dimensions should be sparse. Got nDimI: %ld, d1: %ld, d2: %ld", nDimI, d1, d2);
  THCIndexTensor *indices = THCSTensor_(newIndices)(state, self);
  int64_t nnz = THCSTensor_(nnz)(state, self);
  THCIndexTensor *buffer = THCIndexTensor_(newWithSize1d)(state, nnz);
  THCIndexTensor *slice1 = THCIndexTensor_(newSelect)(state, indices, 0, d1);
  THCIndexTensor *slice2 = THCIndexTensor_(newSelect)(state, indices, 0, d2);
  THCIndexTensor_(copy)(state, buffer, slice1);
  THCIndexTensor_(copy)(state, slice1, slice2);
  THCIndexTensor_(copy)(state, slice2, buffer);
  int64_t i = self->size[d1];
  self->size[d1] = self->size[d2];
  self->size[d2] = i;
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
