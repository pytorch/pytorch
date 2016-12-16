#include "THCSTensor.h"
#include "THCApply.cuh"
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
    for (IndexType d = 0; d < indices.sizes[0]; d++)
      index = other.sizes[d] * index + indices.data[d * indskip + linearId];
    other.data[index] = other.data[index] + values.data[linearId];
  }
}

#include "generic/THCSTensor.cu"
#include "THCSGenerateAllTypes.h"

#include "generic/THCSTensorMath.cu"
#include "THCSGenerateAllTypes.h"
