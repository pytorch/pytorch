#pragma once

#include <ATen/CUDAGeneratorImpl.h>

namespace at {
namespace cuda {
namespace philox {

// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks philox_kernelarg_t in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a free function here, in ATen/cuda.
// Any cuda consumer can include this header.
__device__ __forceinline__ std::pair<uint64_t, uint64_t> unpack(at::philox_kernelarg_t arg) {
  if (arg.has_device_ptrs_) {
    return std::make_pair(*(arg.state_ptrs_.first), *(arg.state_ptrs_.second));
  } else {
    return arg.state_;
  }
}

}
}
}
