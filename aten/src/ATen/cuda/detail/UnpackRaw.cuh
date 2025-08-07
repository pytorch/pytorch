// No "#pragma once" because this is a raw definition that can be copied by jit codegen.
// Eager mode clients should not include this file directly, instead,
// they should #include <ATen/cuda/PhiloxUtils.cuh>, which has a #pragma once.

namespace at::cuda::philox {

// In-kernel call to retrieve philox seed and offset from a PhiloxCudaState instance whether
// that instance was created with graph capture underway or not.
// See Note [CUDA Graph-safe RNG states].
//
// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks PhiloxCudaState in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define a __device__ unpack helper here, in ATen/cuda.
//
// The raw definition lives in its own file so jit codegen can easily copy it.
__host__ __device__ __forceinline__ std::tuple<uint64_t, uint64_t>
unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to "unsigned long".
    // *(arg.offset_.ptr) is a broadcast load of a single int64_t to the entire kernel.
    // For most threads' reads it will hit in cache, so it shouldn't hurt performance.
    return std::make_tuple(static_cast<uint64_t>(*arg.seed_.ptr), static_cast<uint64_t>(*(arg.offset_.ptr) + arg.offset_intragraph_));
  } else {
    return std::make_tuple(arg.seed_.val, arg.offset_.val);
  }
}

// Adapted from TE
// extract seed and offset from PhiloxCudaState
__global__ void unpack_cudnn(at::PhiloxCudaState arg, int64_t* seed_ptr, int64_t* offset_ptr);

void unpack_cudnn_wrapper(at::PhiloxCudaState arg, int64_t* seed_ptr, int64_t* offset_ptr, cudaStream_t stream);

} // namespace at::cuda::philox
