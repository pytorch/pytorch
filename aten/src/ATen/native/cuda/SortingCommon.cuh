#pragma once
#include <ATen/core/TensorBase.h>
#include <ATen/ceil_div.h>
#include <ATen/NumericUtils.h>
#include <c10/macros/Macros.h>
#include <stdlib.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>

namespace at {
namespace native {

// Is this questionable namespace pollution?
#if defined(USE_ROCM)
constexpr int MAX_BLOCK_SIZE = 256;

#else
constexpr int MAX_BLOCK_SIZE = 1024;
#endif

// Maximum size per grid dimension that we assume (compute capability >= 2.0)
constexpr int64_t MAX_GRID_SIZE = 65535LL;

static bool getGridFromTiles(int64_t gridTiles, dim3& grid) {
  if (gridTiles > MAX_GRID_SIZE * MAX_GRID_SIZE * MAX_GRID_SIZE) {
    return false;
  }

  int64_t gridX = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
  int64_t gridY = 1;
  int64_t gridZ = 1;

  if (gridTiles > MAX_GRID_SIZE) {
    gridTiles = ceil_div(gridTiles, MAX_GRID_SIZE);
    gridY = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;

    if (gridTiles > MAX_GRID_SIZE) {
      gridTiles = ceil_div(gridTiles, MAX_GRID_SIZE);
      gridZ = gridTiles > MAX_GRID_SIZE ? MAX_GRID_SIZE : gridTiles;
    }
  }

  grid = dim3(gridX, gridY, gridZ);
  return true;
}

template <typename scalar_t, bool handleNaN = false>
struct GTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && at::_isnan(lhs) && !at::_isnan(rhs)) || (lhs > rhs);
  }
};

template <typename scalar_t, bool handleNaN = false>
struct LTOp {
  __device__ bool operator()(const scalar_t& lhs, const scalar_t& rhs) const {
    return (handleNaN && at::_isnan(rhs) && !at::_isnan(lhs)) || (lhs < rhs);
  }
};

template <typename index_t>
__device__ __forceinline__ index_t getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x +
      blockIdx.x;
}

// For slice sorting in Thrust; extracts a slice index from a linear
// index and uses that for comparison
struct SliceComp {
  SliceComp(int64_t size) : sliceSize(size) {}

  __device__ bool operator()(const int64_t& a, const int64_t& b) const {
    // Since the slices are guaranteed to be innermost,
    // the segment is just via int64_t division
    int64_t segA = a / sliceSize;
    int64_t segB = b / sliceSize;
    return segA < segB;
  }

  const int64_t sliceSize;
};

// For sorting in Thurst; extracts a within-slice index from a linear index
struct GlobalIndexToPerSliceIndex {
  GlobalIndexToPerSliceIndex(int64_t size) : sliceSize(size) {}

  __device__ inline void operator()(int64_t& v) const {
    v = v % sliceSize;
  }

  const int64_t sliceSize;
};

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
static uint64_t nextHighestPowerOf2(uint64_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
#ifndef _MSC_VER
  n |= n >> 32;
#endif
  n++;

  return n;
}


// WARNING: This function assumes input tensors are contiguous
template <typename scalar_t, typename index_t, typename Launcher>
void run_launcher(
    const TensorBase &values,
    const TensorBase &indices,
    const TensorBase &self,
    int64_t dim,
    Launcher l) {
  auto self_info = cuda::detail::getTensorInfo<scalar_t, index_t>(self);
  auto values_info = cuda::detail::getTensorInfo<scalar_t, index_t>(values);
  auto indices_info = cuda::detail::getTensorInfo<int64_t, index_t>(indices);

  int64_t slice_size = self.size(dim);
  /* We use these structures solely to find the offset to */
  /* each slice we are operating on */
  self_info.reduceDim(dim);
  values_info.reduceDim(dim);
  indices_info.reduceDim(dim);

  /* Collapse all other dims */
  int collapse_self_dim = self_info.collapseDims(dim);
  int collapse_values_dim = values_info.collapseDims(dim);
  int collapse_indices_dim = indices_info.collapseDims(dim);

  int64_t num_slices = 1;
  for (int i = 0; i < self_info.dims; ++i) {
    num_slices *= self_info.sizes[i];
  }

  /* This is used as a template parameter to calculate indices. */
  /* We only specialize it if all collapsed dim sizes are the */
  /* same; otherwise, we use -1 which is the specialization */
  /* parameter for arbitrary dimensions */
  int all_dims = self_info.dims;
  if (values_info.dims != all_dims || indices_info.dims != all_dims) {
    all_dims = -1;
  }

  if (all_dims == 1) {
    l.template launch<scalar_t, index_t, 1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 2) {
    l.template launch<scalar_t, index_t, 2>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else if (all_dims == 3) {
    l.template launch<scalar_t, index_t, 3>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  } else {
    l.template launch<scalar_t, index_t, -1>(
        values_info,
        collapse_values_dim,
        indices_info,
        collapse_indices_dim,
        self_info,
        collapse_self_dim,
        num_slices,
        slice_size);
  }
}

} // namespace native
} // namespace at
