#pragma once

#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAGuard.h>

namespace at {
namespace cuda {
namespace philox {

// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks PhiloxCudaState in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a free function here, in ATen/cuda.
// Any cuda consumer can include this header.
__device__ __forceinline__ std::tuple<uint64_t, uint64_t, uint64_t>
unpack(at::PhiloxCudaState arg) {
  if (arg.has_device_ptrs_) {
    uint64_t seed = *arg.seed_ptr_;
    uint64_t offset = *arg.offset_ptr_;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
        blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
      *arg.next_offset_ptr_ = offset + arg.increment_;
    }
    return {seed, arg.subseq_pool_start_, offset};
  } else {
    return {arg.seed_, arg.subseq_pool_start_, arg.offset_};
  }
}

} // namespace philox
} // namespace cuda
} // namespace at
