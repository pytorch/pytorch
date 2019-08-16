#include <ATen/ATen.h>

namespace {

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
#ifdef __HIP_PLATFORM_HCC__
C10_LAUNCH_BOUNDS_1(512)
#endif
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

    int64_t indexValue = index.data[indexOffset];
    assert(indexValue >= 0 && indexValue < src.sizes[dim]);
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

}  // namespace


#define RUN(TYPE, DIMS, REAL)                                           \
  THCudaTensor_gatherKernel<TYPE, REAL, DIMS>                                \
  <<<grid, block, 0, THCState_getCurrentStreamOnDevice(state, curDevice)>>>(               \
    tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THCTensor_(gather)(THCState* state, THCTensor *tensor,
                         THCTensor *src, int dim, THCudaLongTensor *index) {
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 2, tensor, src));
  THCAssertSameGPU(THCudaLongTensor_checkGPU(state, 1, index));

  THArgCheck(THCudaLongTensor_nDimensionLegacyNoScalars(state, index) == THCTensor_(nDimensionLegacyNoScalars)(state, src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(tensor->sizes().equals(index->sizes()), 4,
             "Index tensor must have the same size as output tensor.");
  THArgCheck(dim >= 0 && dim < THCTensor_(nDimensionLegacyNoScalars)(state, tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, src) == THCTensor_(nDimensionLegacyNoScalars)(state, tensor), 2,
             "Input tensor must have same dimensions as output tensor");

  for (int d = 0; d < THCTensor_(nDimensionLegacyNoScalars)(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCTensor_(sizeLegacyNoScalars)(state, tensor, d) == THCTensor_(sizeLegacyNoScalars)(state, src, d), 2,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THCTensor_(nDimensionLegacyNoScalars)(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);


  const ptrdiff_t totalElements = THCudaLongTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  THArgCheck(getApplyGrid(state, totalElements, grid, curDevice), 1, CUTORCH_DIM_WARNING);

  THCTensor* oldTensor = NULL;
  if (THCTensor_maybeOverlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCTensor_(newContiguous)(state, tensor);
  }

  if (totalElements > 0) {
    if (THCTensor_canUse32BitIndexMath(state, tensor) &&
        THCTensor_canUse32BitIndexMath(state, src) &&
        THCTensor_canUse32BitIndexMath(state, index)) {
      TensorInfo<scalar_t, unsigned int> tensorInfo =
        getTensorInfo<scalar_t, THCTensor, unsigned int>(state, tensor);
      TensorInfo<scalar_t, unsigned int> srcInfo =
        getTensorInfo<scalar_t, THCTensor, unsigned int>(state, src);
      TensorInfo<int64_t, unsigned int> indexInfo =
        getTensorInfo<int64_t, THCudaLongTensor, unsigned int>(state, index);

      // Specialize for a small number of dimensions.
      switch (indexInfo.dims) {
        case 1:
          RUN(unsigned int, 1, scalar_t);
          THCudaCheck(cudaGetLastError());
          break;
        case 2:
          RUN(unsigned int, 2, scalar_t);
          THCudaCheck(cudaGetLastError());
          break;
        case 3:
          RUN(unsigned int, 3, scalar_t);
          THCudaCheck(cudaGetLastError());
          break;
        default:
          RUN(unsigned int, -1, scalar_t);
          THCudaCheck(cudaGetLastError());
          break;
      }
    } else {
      TensorInfo<scalar_t, uint64_t> tensorInfo =
        getTensorInfo<scalar_t, THCTensor, uint64_t>(state, tensor);
      TensorInfo<scalar_t, uint64_t> srcInfo =
        getTensorInfo<scalar_t, THCTensor, uint64_t>(state, src);
      TensorInfo<int64_t, uint64_t> indexInfo =
        getTensorInfo<int64_t, THCudaLongTensor, uint64_t>(state, index);
      RUN(uint64_t, -1, scalar_t);
      THCudaCheck(cudaGetLastError());
    }
  }

  if (oldTensor) {
    THCTensor_copyIgnoringOverlaps<scalar_t>(state, oldTensor, tensor);
    THCTensor_(free)(state, tensor);
    tensor = oldTensor;
  }
  THCudaCheck(cudaGetLastError());
}

#undef RUN

namespace at { namespace native {

Tensor & gather_out_cuda(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather_out(result, self, dim, index);
}

Tensor gather_cuda(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad) {
  return legacy::cpu::_th_gather(self, dim, index);
}

}}  // namespace at::native