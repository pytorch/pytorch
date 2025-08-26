#include <ATen/cuda/CUDAContext.h>

#include <cuda_runtime.h>

namespace at::cuda {

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T ATenCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

namespace {

// Threads per block for our apply kernel
// FIXME: use occupancy calculator instead
constexpr uint32_t AT_APPLY_THREADS_PER_BLOCK = 512;
constexpr uint32_t AT_APPLY_BLOCKS_PER_SM = 4;

template <int step = 1>
inline bool getApplyGrid(uint64_t totalElements, dim3& grid, c10::DeviceIndex curDevice, int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK) {
  if (curDevice == -1) return false;
  uint64_t numel_per_thread = static_cast<uint64_t>(max_threads_per_block) * static_cast<uint64_t>(step);
  uint64_t numBlocks = ATenCeilDiv(totalElements, numel_per_thread);
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
    numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

constexpr int getApplyBlocksPerSM() {
  return AT_APPLY_BLOCKS_PER_SM;
}

constexpr int getApplyBlockSize() {
  return AT_APPLY_THREADS_PER_BLOCK;
}

inline dim3 getApplyBlock(int max_threads_per_block=AT_APPLY_THREADS_PER_BLOCK) {
  return dim3(max_threads_per_block);
}

} // anonymous namespace
} // namespace at::cuda
