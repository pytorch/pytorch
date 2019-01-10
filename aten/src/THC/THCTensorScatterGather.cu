#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCAtomics.cuh"
#include "THCApply.cuh"

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, typename Real, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

// Same as above but using a dynamic number of dimensions.
template <typename IndexType, typename Real>
struct IndexToScatterGatherOffsets<IndexType, Real, -1> {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_gatherKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset,
                                                          src, &srcOffset);

    int64_t indexValue = index.data[indexOffset] - TH_INDEX_BASE;
    assert(indexValue >= 0 && indexValue < src.sizes[dim]);
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_scatterKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          src, &srcOffset,
                                                          tensor, &tensorOffset);

    int64_t indexValue = index.data[indexOffset] - TH_INDEX_BASE;
    assert(indexValue >= 0 && indexValue < tensor.sizes[dim]);
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_scatterAddKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<Real, IndexType> src,
    TensorInfo<int64_t, IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          src, &srcOffset,
                                                          tensor, &tensorOffset);

    int64_t indexValue = index.data[indexOffset] - TH_INDEX_BASE;
    assert(indexValue >= 0 && indexValue < tensor.sizes[dim]);
    tensorOffset += indexValue * tensor.strides[dim];

    atomicAdd(&tensor.data[tensorOffset], src.data[srcOffset]);
  }
}

template <typename IndexType, typename Real, int Dims>
__global__ void THCudaTensor_scatterFillKernel(
    TensorInfo<Real, IndexType> tensor,
    TensorInfo<int64_t, IndexType> index,
    Real value,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Real, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset);

    int64_t indexValue = index.data[indexOffset] - TH_INDEX_BASE;
    assert(indexValue >= 0 && indexValue < tensor.sizes[dim]);
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = value;
  }
}

#include "generic/THCTensorScatterGather.cu"
#include "THCGenerateAllTypes.h"
