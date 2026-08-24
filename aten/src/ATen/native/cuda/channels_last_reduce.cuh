#pragma once

// Shared building blocks for reductions over a channels-last tensor viewed as
// [reduction, stride], where the reduced axis is the outer one. blockDim.x
// spans the stride (channels) so a warp's global reads stay contiguous, while
// blockDim.y and gridDim.y split the reduced axis, keeping the grid wide even
// when the stride is small. Blocks in a gridDim.y slice stage their partial
// results and the last one to arrive combines them.
//
// Used by the batch norm and group norm channels-last kernels.

#include <ATen/ceil_div.h>
#include <ATen/native/cuda/LaunchUtils.h>
#include <c10/macros/Macros.h>

#include <algorithm>

namespace at::native {

// The maximum number of threads in a block.
// Note SortingCommon.cuh separately defines an at::native::MAX_BLOCK_SIZE with a
// different value; no translation unit includes both today, but a future one
// would have to reconcile them.
#if defined(USE_ROCM)
constexpr int MAX_BLOCK_SIZE = 1024;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

constexpr int ELEMENTS_PER_ITER =
    4; // enables concurrency within each thread to hide latency
constexpr int ELEMENTS_PER_THREAD = 16;
constexpr int OPTIMAL_TILE_W = 32;
constexpr int MAX_H_BLOCK = 128;

__host__ inline void flexible_launch_configs(
    const int reduction,
    const int stride,
    dim3& block,
    dim3& grid,
    const bool coop_flag = false) {
  int block_x = std::min(lastPow2(stride), OPTIMAL_TILE_W);
  int block_y = std::min(
      lastPow2(at::ceil_div(reduction, ELEMENTS_PER_THREAD)),
      MAX_BLOCK_SIZE / block_x);
  if (block_x * block_y != MAX_BLOCK_SIZE) {
    block_x = std::min(lastPow2(stride), MAX_BLOCK_SIZE / block_y);
  }

  int grid_x = at::ceil_div(stride, block_x);
  int grid_y =
      std::min(at::ceil_div(reduction, block_y * ELEMENTS_PER_THREAD),
               MAX_H_BLOCK);
  if (coop_flag) {
    // it's not worth having a grid reduction if the reduction dimension is not
    // big enough
    grid_y = grid_y < 8 ? 1 : grid_y;
  }

  block.x = block_x;
  block.y = block_y;
  block.z = 1;
  grid.x = grid_x;
  grid.y = grid_y;
  grid.z = 1;
}

template <typename T, typename C>
__device__ __forceinline__ void welford_merge_element(
    C& count,
    T& mean,
    T& m2n,
    const C& count_new,
    const T& mean_new,
    const T& m2n_new) {
  T factor = T(1.0) / ::max(C(1), (count + count_new));
  T delta0 = mean - mean_new;
  mean = (mean_new * count_new + mean * count) * factor;
  m2n += m2n_new + delta0 * delta0 * count_new * count * factor;
  count += count_new;
}

// merge mean/m2n among threadIdx.y within block
template <typename T, typename C>
__device__ __forceinline__ void welford_merge_block_vertical(
    C& count,
    T& mean,
    T& m2n,
    C* shmem_count,
    T* shmem_mean,
    T* shmem_m2n) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset * 2) {
      shmem_mean[address_base] = mean;
      shmem_m2n[address_base] = m2n;
      shmem_count[address_base] = count;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;
      // read shared memory back to register for reduction
      auto count_new = shmem_count[address];
      auto mean_new = shmem_mean[address];
      auto m2n_new = shmem_m2n[address];

      welford_merge_element(count, mean, m2n, count_new, mean_new, m2n_new);
    }
  }
}

// merge a pair of sums among threadIdx.y within block
template <typename T>
__device__ __forceinline__ void merge_block_vertical_backward(
    T& sum_dy,
    T& sum_dy_xmu,
    T* shmem_sum_dy,
    T* shmem_sum_dy_xmu) {
  // write to shared memory
  auto address_base = threadIdx.x + threadIdx.y * blockDim.x;

#pragma unroll
  for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
    if (threadIdx.y < offset * 2) {
      shmem_sum_dy[address_base] = sum_dy;
      shmem_sum_dy_xmu[address_base] = sum_dy_xmu;
    }
    __syncthreads();
    if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
      auto address = address_base + offset * blockDim.x;

      sum_dy += shmem_sum_dy[address];
      sum_dy_xmu += shmem_sum_dy_xmu[address];
    }
  }
}

} // namespace at::native
