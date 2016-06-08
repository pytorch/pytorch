#include "THCTensorMath.h"
#include "THCGeneral.h"
#include "THCApply.cuh"

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<float, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<float, IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<float, IndexType>& t2, IndexType* t2Offset) {
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
      const TensorInfo<float, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<float, IndexType>& t2, IndexType* t2Offset) {
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
      const TensorInfo<float, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<float, IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<float, IndexType>& t2, IndexType* t2Offset) {
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
      const TensorInfo<float, IndexType>& index, IndexType* indexOffset,
      const TensorInfo<float, IndexType>& t2, IndexType* t2Offset) {
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
    TensorInfo<float, IndexType> tensor,
    TensorInfo<float, IndexType> src,
    TensorInfo<float, IndexType> index,
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

  THArgCheck(THCudaTensor_nDimension(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);


  const long totalElements = THCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, totalElements, grid), 1, CUTORCH_DIM_WARNING);

  THCudaTensor* oldTensor = NULL;
  if (TensorUtils<THCudaTensor>::overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCudaTensor_newContiguous(state, tensor);
  }

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, tensor) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, index)) {
    TensorInfo<float, unsigned int> tensorInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, tensor);
    TensorInfo<float, unsigned int> srcInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, src);
    TensorInfo<float, unsigned int> indexInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, index);

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
    TensorInfo<float, unsigned long> tensorInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, tensor);
    TensorInfo<float, unsigned long> srcInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, src);
    TensorInfo<float, unsigned long> indexInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, index);

    RUN(unsigned long, -1)
  }

  if (oldTensor) {
    TensorUtils<THCudaTensor>::copyIgnoringOverlaps(state, oldTensor, tensor);
    THCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN


template <typename IndexType, int Dims>
__global__ void THCudaTensor_scatterKernel(
    TensorInfo<float, IndexType> tensor,
    TensorInfo<float, IndexType> src,
    TensorInfo<float, IndexType> index,
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

  THArgCheck(THCudaTensor_nDimension(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);

  const long totalElements = THCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, totalElements, grid), 1, CUTORCH_DIM_WARNING);

  THCudaTensor* oldTensor = NULL;
  if (TensorUtils<THCudaTensor>::overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCudaTensor_newContiguous(state, tensor);
  }

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, tensor) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, src) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, index)) {
    TensorInfo<float, unsigned int> tensorInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, tensor);
    TensorInfo<float, unsigned int> srcInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, src);
    TensorInfo<float, unsigned int> indexInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, index);

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
    TensorInfo<float, unsigned long> tensorInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, tensor);
    TensorInfo<float, unsigned long> srcInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, src);
    TensorInfo<float, unsigned long> indexInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, index);

    RUN(unsigned long, -1)
  }

  if (oldTensor) {
    TensorUtils<THCudaTensor>::copyIgnoringOverlaps(state, oldTensor, tensor);
    THCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN


template <typename IndexType, int Dims>
__global__ void THCudaTensor_scatterFillKernel(
    TensorInfo<float, IndexType> tensor,
    TensorInfo<float, IndexType> index,
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

void
THCudaTensor_scatterFill(THCState* state, THCudaTensor *tensor,
                         int dim, THCudaTensor *index, float value) {
  THAssert(THCudaTensor_checkGPU(state, 2, tensor, index));

  THArgCheck(dim >= 0 && dim < THCudaTensor_nDimension(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THCudaTensor_nDimension(state, index) ==
             THCudaTensor_nDimension(state, tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  for (int d = 0; d < THCudaTensor_nDimension(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THCudaTensor_size(state, tensor, d) ==
                 THCudaTensor_size(state, index, d), 4,
                 "Index tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  THArgCheck(THCudaTensor_nDimension(state, tensor) <= MAX_CUTORCH_DIMS,
             1, CUTORCH_DIM_WARNING);

  const long totalElements = THCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  THArgCheck(getApplyGrid(state, totalElements, grid), 1, CUTORCH_DIM_WARNING);

  THCudaTensor* oldTensor = NULL;
  if (TensorUtils<THCudaTensor>::overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THCudaTensor_newContiguous(state, tensor);
  }

  if (TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, tensor) &&
      TensorUtils<THCudaTensor>::canUse32BitIndexMath(state, index)) {
    TensorInfo<float, unsigned int> tensorInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, tensor);
    TensorInfo<float, unsigned int> indexInfo =
      getTensorInfo<THCudaTensor, unsigned int>(state, index);

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
    TensorInfo<float, unsigned long> tensorInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, tensor);
    TensorInfo<float, unsigned long> indexInfo =
      getTensorInfo<THCudaTensor, unsigned long>(state, index);

    RUN(unsigned long, -1);
  }

  if (oldTensor) {
    TensorUtils<THCudaTensor>::copyIgnoringOverlaps(state, oldTensor, tensor);
    THCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN
