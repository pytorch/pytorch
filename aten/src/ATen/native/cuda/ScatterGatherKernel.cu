#include <THC/THCTensorMath.h>
#include <THC/THCGeneral.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCApply.cuh>

#include <ATen/cuda/CUDAApplyUtils.cuh>

namespace {

using at::cuda::detail::TensorInfo;

// TODO: Below structs have been copied from THC/THCTensorScatterGather.cu.
// Once the CUDA scatter_add and gather operations are ported to ATen, that file
  // should be deleted.
  
  // Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, typename Real, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      IndexType linearId, const int64_t dim,
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
      IndexType linearId, const int64_t dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
    for (int64_t d = Dims - 1; d >= 0; d--) {
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
      IndexType linearId, const int64_t dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
    for (int64_t d = index.dims - 1; d >= 0; d--) {
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
      IndexType linearId, const int64_t dim,
      const TensorInfo<int64_t, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<Real, IndexType>& t2, IndexType* t2Offset) {
    for (int64_t d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

template <typename index_t, typename scalar_t, int dims>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void scatter_kernel(
  TensorInfo<scalar_t, index_t> self,
  TensorInfo<scalar_t, index_t> src,
  TensorInfo<int64_t, index_t> index,
  const int64_t dim,
  const index_t total_elements) {

  for (index_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < total_elements;
       linearId += gridDim.x * blockDim.x) {
    index_t tensorOffset = 0;
    index_t srcOffset = 0;
    index_t indexOffset = 0;

    IndexToScatterGatherOffsets<index_t, scalar_t, dims>::compute(linearId, dim,
                                                                index, &indexOffset,
                                                                src, &srcOffset,
                                                                self, &tensorOffset);

    int64_t indexValue = index.data[indexOffset];
    CUDA_KERNEL_ASSERT(indexValue >= 0 && indexValue < self.sizes[dim]);
    tensorOffset += indexValue * self.strides[dim];

    self.data[tensorOffset] = src.data[srcOffset];
  }
}

template <typename index_t, typename scalar_t, int dims>
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
__global__ void scatter_fill_kernel(
    TensorInfo<scalar_t, index_t> self,
    scalar_t value,
    TensorInfo<int64_t, index_t> index,
    const int64_t dim,
    const index_t totalElements) {
  for (index_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    index_t tensorOffset = 0;
    index_t indexOffset = 0;

    IndexToScatterGatherOffsets<index_t, scalar_t, dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          self, &tensorOffset);

    int64_t indexValue = index.data[indexOffset];
    CUDA_KERNEL_ASSERT(indexValue >= 0 && indexValue < self.sizes[dim]);
    tensorOffset += indexValue * self.strides[dim];

    self.data[tensorOffset] = value;
  }
}

  
} // anonymous namespace
