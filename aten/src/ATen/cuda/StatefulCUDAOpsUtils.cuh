#pragma once

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAGuard.h>

namespace at {
namespace cuda {

c10::optional<c10::Stream> stateUpdateStream(DeviceIndex device_index);

namespace philox {

// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks philox_kernelarg_t in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a free function here, in ATen/cuda.
// Any cuda consumer can include this header.
__device__ __forceinline__ std::pair<uint64_t, uint64_t> unpack(at::philox_kernelarg_t arg) {
  if (arg.has_device_ptrs_) {
    uint64_t seed = *args.seed_ptr_this_launch_;
    uint64_t offset = *arg.offset_ptr_this_launch_;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      *offset_next_launch = offset + args.increment;
    }
    return std::make_pair(seed, offset);
  } else {
    return arg.state_;
  }
}

} // namespace philox
} // namespace cuda
} // namespace at
