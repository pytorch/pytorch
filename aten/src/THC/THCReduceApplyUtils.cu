#include <THC/THCReduceApplyUtils.cuh>

#include <assert.h>
#include <stdlib.h>

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
#define MAX_GRID_SIZE 65535LL

void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg) {
  int64_t dims = THCudaTensor_nDimensionLegacyAll(state, tensor);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, arg, CUTORCH_DIM_WARNING);
}

bool THC_getGridFromTiles(ptrdiff_t gridTiles, dim3& grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  int64_t gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  int64_t gridY = 1;
  int64_t gridZ = 1;

  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = THCCeilDiv(gridTiles, (ptrdiff_t) MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = THCCeilDiv(gridTiles, (ptrdiff_t) MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }

  grid = dim3(gridX, gridY, gridZ);
  return true;
}
