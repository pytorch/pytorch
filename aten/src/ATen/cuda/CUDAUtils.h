#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace at { namespace cuda {

template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

// Threads per block for various kernels
// FIXME: use occupancy calculator instead
constexpr uint32_t AT_APPLY_THREADS_PER_BLOCK = 512;
constexpr uint32_t AT_APPLY_BLOCKS_PER_SM = 4;

inline dim3 getApplyBlock() {
  return dim3(AT_APPLY_THREADS_PER_BLOCK);
}

template <int step = 1>
inline bool getApplyGrid(uint64_t totalElements, dim3& grid, int64_t curDevice) {
  if (curDevice == -1) return false;
  uint64_t numel_per_thread = static_cast<uint64_t>(AT_APPLY_THREADS_PER_BLOCK) * static_cast<uint64_t>(step);
  uint64_t numBlocks = ATenCeilDiv(totalElements, numel_per_thread);
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
      numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

// Check if every tensor in a list of tensors matches the current
// device.
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kCUDA, current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice) return false;
  }
  return true;
}

}} // namespace at::cuda
