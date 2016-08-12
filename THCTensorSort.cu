#include "THCReduceApplyUtils.cuh"
#include "THCSortUtils.cuh"
#include "THCTensorCopy.h"
#include "THCTensorTypeUtils.cuh"

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

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
unsigned long nextHighestPowerOf2(unsigned long n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;

  return n;
}

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
template <typename IndexType, int Dim>
__global__ void
fillSliceWithIndex(TensorInfo<long, IndexType> out,
                   IndexType totalSlices,
                   IndexType sliceSize,
                   IndexType sliceStride) {
  IndexType slice = getLinearBlockId<IndexType>();

  if (slice >= totalSlices) {
    return;
  }

  const unsigned long offset =
    IndexToOffset<long, IndexType, Dim>::get(slice, out);
  long* base = &out.data[offset];

  for (long i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    // Torch indices are 1-based (hence the +1)
    base[i * sliceStride] = i + 1;
  }
}

void THCudaLongTensor_fillSliceWithIndex(THCState* state,
                                         THCudaLongTensor* t,
                                         int dim) {
  long dims = THCudaLongTensor_nDimension(state, t);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, 2, CUTORCH_DIM_WARNING);

  long inElements = THCudaLongTensor_nElement(state, t);
  long sliceSize = THCudaLongTensor_size(state, t, dim);
  long numSlices = inElements / sliceSize;

  dim3 grid;
  if (!THC_getGridFromTiles(numSlices, grid)) {
    THError("Slice to fill with indices is too large");
  }

  long maxThreads =
    THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock;
  long numThreads = sliceSize;
  if (numThreads > maxThreads) {
    numThreads = maxThreads;
  }

  dim3 block(numThreads);

#define FILL_INDEX(T, DIM)                                       \
  fillSliceWithIndex<T, DIM>                                     \
    <<<grid, block, 0, THCState_getCurrentStream(state)>>>(      \
      info, numSlices, sliceSize, info.strides[collapseDim])

  if (TensorUtils<THCudaLongTensor>::canUse32BitIndexMath(state, t)) {
    TensorInfo<long, unsigned int> info =
      getTensorInfo<THCudaLongTensor, unsigned int>(state, t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    if (info.isContiguous()) {
      FILL_INDEX(unsigned int, -2);
    } else {
      if (info.dims == 1) {
        FILL_INDEX(unsigned int, 1);
      } else if (info.dims == 2) {
        FILL_INDEX(unsigned int, 2);
      } else {
        FILL_INDEX(unsigned int, -1);
      }
    }
  } else {
    TensorInfo<long, unsigned long> info =
      getTensorInfo<THCudaLongTensor, unsigned long>(state, t);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    // catch-all implementation
    FILL_INDEX(unsigned long, -1);
  }

#undef FILL_INDEX

  THCudaCheck(cudaGetLastError());
}

// For slice sorting in Thrust; extracts a slice index from a linear
// index and uses that for comparison
struct SliceComp {
  SliceComp(long size) : sliceSize(size) {}

  __device__ bool operator()(const long& a, const long& b) const {
    // Since the slices are guaranteed to be innermost, the segment is
    // just via long division
    long segA = a / sliceSize;
    long segB = b / sliceSize;
    return segA < segB;
  }

  const long sliceSize;
};

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  GlobalIndexToPerSliceIndex(long size) : sliceSize(size) {}

  __device__ inline void operator()(long& v) const {
    v = v % sliceSize + 1;
  }

  const long sliceSize;
};

#include "generic/THCTensorSort.cu"
#include "THCGenerateAllTypes.h"
