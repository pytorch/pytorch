#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCApply.cuh"


// Similar to TensorInfo as defined in THCReduceApplyUtils.h, but it preserves
// the exact dimensionality of the tensor instead of flattening contiguous or
// size-1 dimensions. This is required for scatter/gather kernels because we
// need to know the indices along all dimensions.
template <typename IndexType>
struct SimpleTensorInfo {
  SimpleTensorInfo(THCState* state, THCudaTensor* t) {
    data = THCudaTensor_data(state, t);
    dims = THCudaTensor_nDimension(state, t);
    for (int d = 0; d < dims; d++) {
      sizes[d] = THCudaTensor_size(state, t, d);
      strides[d] = THCudaTensor_stride(state, t, d);
    }
  }

  float* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};


// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const SimpleTensorInfo<IndexType>& index, IndexType* indexOffset,
      const SimpleTensorInfo<IndexType>& t1, IndexType* t1Offset,
      const SimpleTensorInfo<IndexType>& t2, IndexType* t2Offset) {
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
      const SimpleTensorInfo<IndexType>& index, IndexType* indexOffset,
      const SimpleTensorInfo<IndexType>& t2, IndexType* t2Offset) {
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
template <typename IndexType>
struct IndexToScatterGatherOffsets<IndexType, -1> {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const SimpleTensorInfo<IndexType>& index, IndexType* indexOffset,
      const SimpleTensorInfo<IndexType>& t1, IndexType* t1Offset,
      const SimpleTensorInfo<IndexType>& t2, IndexType* t2Offset) {
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
      const SimpleTensorInfo<IndexType>& index, IndexType* indexOffset,
      const SimpleTensorInfo<IndexType>& t2, IndexType* t2Offset) {
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


template <typename IndexType, int Dims>
__global__ void THCudaTensor_gatherKernel(
    SimpleTensorInfo<IndexType> tensor,
    SimpleTensorInfo<IndexType> src,
    SimpleTensorInfo<IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset,
                                                          src, &srcOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset] - 1;
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

#define RUN(TYPE, DIMS)                                              \
  THCudaTensor_gatherKernel<TYPE, DIMS>                              \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(        \
          tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THCudaTensor_gather(THCState* state, THCudaTensor *tensor, THCudaTensor *src, int dim, THCudaTensor *index) {
  THAssert(THCudaTensor_checkGPU(state, 3, tensor, src, index));

  THArgCheck(THCudaTensor_nDimension(state, src) == THCudaTensor_nDimension(state, tensor), 2,
             "Input tensor must have same dimensions as output tensor");
  THArgCheck(dim >= 0 && dim < THCudaTensor_nDimension(state, tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THCudaTensor_nDimension(state, index) == THCudaTensor_nDimension(state, src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(THCudaTensor_isSameSizeAs(state, tensor, index), 4,
             "Index tensor must have the same size as output tensor.");

  for (int d = 0; d < THCudaTensor_nDimension(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCudaTensor_size(state, tensor, d) == THCudaTensor_size(state, src, d), 2,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  if (THCudaTensor_nDimension(state, tensor) > MAX_CUTORCH_DIMS) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  const long totalElements = THCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaTensor* oldTensor = NULL;
  if (THC_overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCudaTensor_newContiguous(state, tensor);
  }

  if (THC_canUse32BitIndexMath(state, tensor) &&
      THC_canUse32BitIndexMath(state, src) &&
      THC_canUse32BitIndexMath(state, index)) {
    SimpleTensorInfo<unsigned int> tensorInfo(state, tensor);
    SimpleTensorInfo<unsigned int> srcInfo(state, src);
    SimpleTensorInfo<unsigned int> indexInfo(state, index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1);
        break;
      case 2:
        RUN(unsigned int, 2);
        break;
      case 3:
        RUN(unsigned int, 3);
        break;
      default:
        RUN(unsigned int, -1);
        break;
    }
  } else {
    SimpleTensorInfo<unsigned long> tensorInfo(state, tensor);
    SimpleTensorInfo<unsigned long> srcInfo(state, src);
    SimpleTensorInfo<unsigned long> indexInfo(state, index);

    RUN(unsigned long, -1)
  }

  if (oldTensor) {
    THCudaTensor_copyIgnoringOverlaps(state, oldTensor, tensor);
    THCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN


template <typename IndexType, int Dims>
__global__ void THCudaTensor_scatterKernel(
    SimpleTensorInfo<IndexType> tensor,
    SimpleTensorInfo<IndexType> src,
    SimpleTensorInfo<IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          src, &srcOffset,
                                                          tensor, &tensorOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset] - 1;
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

#define RUN(TYPE, DIMS)                                              \
  THCudaTensor_scatterKernel<TYPE, DIMS>                             \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(        \
          tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THCudaTensor_scatter(THCState* state, THCudaTensor *tensor, int dim, THCudaTensor *index, THCudaTensor *src) {
  THAssert(THCudaTensor_checkGPU(state, 3, tensor, src, index));

  THArgCheck(dim >= 0 && dim < THCudaTensor_nDimension(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THCudaTensor_nDimension(state, index) == THCudaTensor_nDimension(state, src), 3,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(THCudaTensor_nDimension(state, src) == THCudaTensor_nDimension(state, tensor), 4,
             "Input tensor must have same dimensions as output tensor");
  THArgCheck(THCudaTensor_isSameSizeAs(state, src, index), 3,
             "Index tensor must have the same size as input tensor.");

  for (int d = 0; d < THCudaTensor_nDimension(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCudaTensor_size(state, tensor, d) == THCudaTensor_size(state, src, d), 4,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  if (THCudaTensor_nDimension(state, tensor) > MAX_CUTORCH_DIMS) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  const long totalElements = THCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaTensor* oldTensor = NULL;
  if (THC_overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCudaTensor_newContiguous(state, tensor);
  }

  if (THC_canUse32BitIndexMath(state, tensor) &&
      THC_canUse32BitIndexMath(state, src) &&
      THC_canUse32BitIndexMath(state, index)) {
    SimpleTensorInfo<unsigned int> tensorInfo(state, tensor);
    SimpleTensorInfo<unsigned int> srcInfo(state, src);
    SimpleTensorInfo<unsigned int> indexInfo(state, index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1);
        break;
      case 2:
        RUN(unsigned int, 2);
        break;
      case 3:
        RUN(unsigned int, 3);
        break;
      default:
        RUN(unsigned int, -1);
        break;
    }
  } else {
    SimpleTensorInfo<unsigned long> tensorInfo(state, tensor);
    SimpleTensorInfo<unsigned long> srcInfo(state, src);
    SimpleTensorInfo<unsigned long> indexInfo(state, index);

    RUN(unsigned long, -1)
  }

  if (oldTensor) {
    THCudaTensor_copyIgnoringOverlaps(state, oldTensor, tensor);
    THCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN


template <typename IndexType, int Dims>
__global__ void THCudaTensor_scatterFillKernel(
    SimpleTensorInfo<IndexType> tensor,
    SimpleTensorInfo<IndexType> index,
    float value,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset] - 1;
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = value;
  }
}

#define RUN(TYPE, DIMS)                                            \
  THCudaTensor_scatterFillKernel<TYPE, DIMS>                       \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(      \
          tensorInfo, indexInfo, value, dim, (TYPE)totalElements);

void THCudaTensor_scatterFill(THCState* state, THCudaTensor *tensor, int dim, THCudaTensor *index, float value) {
  THAssert(THCudaTensor_checkGPU(state, 2, tensor, index));

  THArgCheck(dim >= 0 && dim < THCudaTensor_nDimension(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THCudaTensor_nDimension(state, index) == THCudaTensor_nDimension(state, tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  for (int d = 0; d < THCudaTensor_nDimension(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCudaTensor_size(state, tensor, d) == THCudaTensor_size(state, index, d), 4,
                 "Index tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  if (THCudaTensor_nDimension(state, tensor) > MAX_CUTORCH_DIMS) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  const long totalElements = THCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaTensor* oldTensor = NULL;
  if (THC_overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCudaTensor_newContiguous(state, tensor);
  }

  if (THC_canUse32BitIndexMath(state, tensor) &&
      THC_canUse32BitIndexMath(state, index)) {
    SimpleTensorInfo<unsigned int> tensorInfo(state, tensor);
    SimpleTensorInfo<unsigned int> indexInfo(state, index);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1);
        break;
      case 2:
        RUN(unsigned int, 2);
        break;
      case 3:
        RUN(unsigned int, 3);
        break;
      default:
        RUN(unsigned int, -1);
        break;
    }
  } else {
    SimpleTensorInfo<unsigned long> tensorInfo(state, tensor);
    SimpleTensorInfo<unsigned long> indexInfo(state, index);

    RUN(unsigned long, -1);
  }

  if (oldTensor) {
    THCudaTensor_copyIgnoringOverlaps(state, oldTensor, tensor);
    THCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN
