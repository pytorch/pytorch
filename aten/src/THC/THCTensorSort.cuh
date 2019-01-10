#ifndef THC_TENSORSORT_CUH
#define THC_TENSORSORT_CUH

#include "THCReduceApplyUtils.cuh"
#include "THCSortUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorTypeUtils.cuh"

#include "THCThrustAllocator.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

template <typename T>
struct ThrustGTOp {
  __device__ bool operator()(const T& lhs, const T& rhs) const {
    return THCNumerics<T>::gt(lhs, rhs);
  }
};

template <typename T>
struct ThrustLTOp {
  __device__ bool operator()(const T& lhs, const T& rhs) const {
    return THCNumerics<T>::lt(lhs, rhs);
  }
};

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename IndexType, int Dim>
__global__ void
fillSliceWithIndex(TensorInfo<int64_t, IndexType> out,
                   IndexType totalSlices,
                   IndexType sliceSize,
                   IndexType sliceStride) {
  IndexType slice = getLinearBlockId<IndexType>();

  if (slice >= totalSlices) {
    return;
  }

  const uint64_t offset =
    IndexToOffset<int64_t, IndexType, Dim>::get(slice, out);
  int64_t* base = &out.data[offset];

  for (int64_t i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = i + TH_INDEX_BASE;
  }
}

// For slice sorting in Thrust; extracts a slice index from a linear
// index and uses that for comparison
struct SliceComp {
  SliceComp(int64_t size) : sliceSize(size) {}

  __device__ bool operator()(const int64_t& a, const int64_t& b) const {
    // Since the slices are guaranteed to be innermost,
    // the segment is just via int64_t division
    int64_t segA = a / sliceSize;
    int64_t segB = b / sliceSize;
    return segA < segB;
  }

  const int64_t sliceSize;
};

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  GlobalIndexToPerSliceIndex(int64_t size) : sliceSize(size) {}

  __device__ inline void operator()(int64_t& v) const {
    v = v % sliceSize + TH_INDEX_BASE;
  }

  const int64_t sliceSize;
};

void THCudaLongTensor_fillSliceWithIndex(THCState* state,
                                         THCudaLongTensor* t,
                                         int dim);
#endif // THC_TENSORSORT_CUH
