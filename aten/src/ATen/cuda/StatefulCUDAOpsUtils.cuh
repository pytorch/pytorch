#pragma once

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAGuard.h>

namespace at {
namespace cuda {
namespace philox {

 constexpr uint64_t max_kernel_threads =  (uint64_t(1) << 40);

// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks PhiloxCudaState in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a free function here, in ATen/cuda.
// Any cuda consumer can include this header.
__device__ __forceinline__ std::tuple<uint64_t, uint64_t, uint64_t>
unpack(at::PhiloxCudaState arg) {
  uint64_t seq_pool_start = (arg.is_on_device_and_seq_pool_id_ & 0x7fffffff) * max_kernel_threads;

  if (arg.is_on_device_and_seq_pool_id_ & 0x80000000) {
    uint64_t seed = *arg.seed_.ptr;
    // offset was already incremented in the philox_cuda_state() call
    uint64_t offset = *arg.offset_.ptr - arg.increment_;
    return std::make_tuple(seed, seq_pool_start, offset);
  } else {
    return std::make_tuple(arg.seed_.val, seq_pool_start, arg.offset_.val);
  }
}

void update_offset(at::PhiloxCudaState arg);

} // namespace philox
} // namespace cuda
} // namespace at
