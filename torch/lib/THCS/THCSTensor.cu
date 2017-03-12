#include "THCSTensor.h"
#include "THCApply.cuh"
#include "THCTensorMathPointwise.cuh"
#include "stdio.h"

template <typename IndexType, typename Real>
__global__ void THCSTensor_toDenseKernel(
    TensorInfo<Real, IndexType> other,
    TensorInfo<long, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
      linearId < nnz;
      linearId += gridDim.x * blockDim.x) {
    IndexType index = 0;
    IndexType indskip = indices.strides[0];
    IndexType valueStride = values.strides[0];
    TensorAddOp<Real> addOp = TensorAddOp<Real>();
    for (IndexType d = 0; d < indices.sizes[0]; d++) {
      index = other.sizes[d] * index + indices.data[d * indskip + linearId];
    }
    for (IndexType k = 0; k < values.strides[0]; k++) {
      addOp(other.data + index * valueStride + k, values.data + linearId * valueStride + k);
    }
  }
}

template <typename IndexType, typename Real>
__global__ void THCSTensor_uniqueValuesReorderKernel(
    TensorInfo<long, IndexType> indices,
    TensorInfo<Real, IndexType> values,
    const IndexType nnz) {
  IndexType i = 0;
  IndexType indskip = indices.strides[0];
  IndexType valueStride = values.strides[0];
  TensorAddOp<Real> addOp = TensorAddOp<Real>();
  for (IndexType j = 1; j < nnz; j++) {
    int cmp = 1;
    for (IndexType d = 0; d < indices.sizes[0]; d++) {
      if (indices.data[d * indskip + i] != indices.data[d * indskip + j]) {
        cmp = 0;
        break;
      }
    }
    if (cmp) {
      for (IndexType k = blockIdx.x * blockDim.x + threadIdx.x;
           k < valueStride;
           k += gridDim.x * blockDim.x) {
        addOp(values.data + i * valueStride + k, values.data + j * valueStride + k);
      }
    } else {
      ++i;
      for (IndexType k = blockIdx.x * blockDim.x + threadIdx.x;
           k < valueStride;
           k += gridDim.x * blockDim.x) {
        values.data[i * valueStride + k] = values.data[j * valueStride + k];
      }
    }
  }
}

template <typename IndexType, typename Real>
__global__ void THCSTensor_uniqueIndicesReorderKernel(
    TensorInfo<long, IndexType> indices,
    const IndexType nnz,
    IndexType* resultNnz) {
  IndexType i = 0;
  IndexType indskip = indices.strides[0];
  for (IndexType j = 1; j < nnz; j++) {
    int cmp = 1;
    for (IndexType d = 0; d < indices.sizes[0]; d++) {
      if (indices.data[d * indskip + i] != indices.data[d * indskip + j]) {
        cmp = 0;
        break;
      }
    }
    if (!cmp) {
      ++i;
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (IndexType d = 0; d < indices.sizes[0]; d++) {
          indices.data[d * indskip + i] = indices.data[d * indskip + j];
        }
      }
    }
  }
  *resultNnz = i + 1;
}

#include "generic/THCSTensor.cu"
#include "THCSGenerateAllTypes.h"

#include "generic/THCSTensorMath.cu"
#include "THCSGenerateAllTypes.h"
