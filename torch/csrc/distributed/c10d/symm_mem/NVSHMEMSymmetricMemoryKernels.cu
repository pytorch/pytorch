#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>
#include <torch/csrc/distributed/c10d/symm_mem/NVSHMEMSymmetricMemoryKernels.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>

namespace c10d::symmetric_memory {

void launch_barrier_kernel(
    uint32_t** signal_pads_dev,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  check_channel(channel, world_size, get_signal_pad_size());
  barrier_kernel<<<
      1,
      std::max(at::cuda::warp_size(), world_size),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      signal_pads_dev, channel, rank, world_size, timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_put_signal_kernel(
    uint32_t** signal_pads_dev,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  check_channel(channel, world_size, get_signal_pad_size());
  put_signal_kernel<<<
      1,
      at::cuda::warp_size(),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      signal_pads_dev, dst_rank, channel, rank, world_size, timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_wait_signal_kernel(
    uint32_t** signal_pads_dev,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms) {
  check_channel(channel, world_size, get_signal_pad_size());
  wait_signal_kernel<<<
      1,
      at::cuda::warp_size(),
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      signal_pads_dev, src_rank, channel, rank, world_size, timeout_ms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace c10d::symmetric_memory
