#include "THCReduceApplyUtils.cuh"

#include <assert.h>
#include <stdlib.h>

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
#define MAX_GRID_SIZE 65535LL

void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg) {
  long dims = THCudaTensor_nDimension(state, tensor);
  THArgCheck(dims <= MAX_CUTORCH_DIMS, arg, CUTORCH_DIM_WARNING);
}

bool THC_getGridFromTiles(ptrdiff_t gridTiles, dim3& grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  long gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  long gridY = 1;
  long gridZ = 1;

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
