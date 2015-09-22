#include "THCReduceApplyUtils.cuh"

#include <assert.h>
#include <stdlib.h>

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
#define MAX_GRID_SIZE 65535L

void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg) {
  long dims = THCudaTensor_nDimension(state, tensor);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, arg, CUTORCH_DIM_WARNING);
}

bool THC_canUse32BitIndexMath(THCState* state, THCudaTensor* t) {
  long elements = THCudaTensor_nElement(state, t);
  if (elements >= UINT_MAX) {
    return false;
  }

  long offset = 0;
  long linearId = elements - 1;

  for (int i = THCudaTensor_nDimension(state, t) - 1; i >= 0; --i) {
    long curDimIndex = linearId % THCudaTensor_size(state, t, i);
    long curDimOffset = curDimIndex * THCudaTensor_stride(state, t, i);
    offset += curDimOffset;
    linearId /= THCudaTensor_size(state, t, i);
  }

  if (offset >= UINT_MAX) {
    return false;
  }

  return true;
}

bool THC_getGridFromTiles(long gridTiles, dim3& grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  long gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  long gridY = 1;
  long gridZ = 1;

  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = THCCeilDiv(gridTiles, MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = THCCeilDiv(gridTiles, MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }

  grid = dim3(gridX, gridY, gridZ);
  return true;
}

namespace {

struct SizeAndStride {
  long size;
  long stride;
};

int compareSizeAndStride(const void* a, const void* b) {
  const SizeAndStride* aS = (const SizeAndStride*) a;
  const SizeAndStride* bS = (const SizeAndStride*) b;

  return aS->stride < bS->stride;
}

}

bool THC_overlappingIndices(THCState* state, THCudaTensor* t) {
  // In this function, we don't care about permutations of the
  // size/stride arrays (transpositions).
  // We order the size/stride arrays by stride, skipping dimensions of
  // size 1. Strides of dimensions of size 1 don't matter, since there
  // is only one addressing point in them.
  // In this reordered view, the tensor is contiguous if
  // stride[dim] == size[dim + 1] * stride[dim + 1] for all `dim`.
  // The tensor has holes if
  // stride[dim] > size[dim + 1] * stride[dim + 1] for one or more
  // `dim`.
  // The tensor has overlaps if
  // stride[dim] < size[dim + 1] * stride[dim + 1] for one or more
  // `dim`, or the innermost stride is 0.

  // Extract size/stride arrays; only consider size >1 dims.
  SizeAndStride info[MAX_CUTORCH_DIMS];

  int dims = THCudaTensor_nDimension(state, t);
  int nonSize1Dims = 0;
  for (int i = 0; i < dims; ++i) {
    long size = THCudaTensor_size(state, t, i);
    if (size > 1) {
      info[nonSize1Dims].size = size;
      info[nonSize1Dims].stride = THCudaTensor_stride(state, t, i);
      ++nonSize1Dims;
    }
  }

  if (nonSize1Dims == 0) {
    // no overlap
    return false;
  }

  // Ascending order (innermost dimension in sorted view is at [0])
  qsort(info, nonSize1Dims, sizeof(SizeAndStride), compareSizeAndStride);

  // Base case: innermost dimension must have stride >= 1
  if (info[nonSize1Dims - 1].stride < 1) {
    return true;
  }

  // Subsequent dimensions, if any
  for (int i = nonSize1Dims - 2; i >= 0; --i) {
    if (info[i].stride < info[i + 1].size * info[i + 1].stride) {
      // There are overlaps
      return true;
    }
  }

  // Tensor has holes or is contiguous
  return false;
}
