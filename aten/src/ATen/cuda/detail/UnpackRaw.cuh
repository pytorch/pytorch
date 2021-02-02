// No "#pragma once" because this is a raw definition that can be copied by jit codegen.
// Eager mode clients should not include this file directly, instead,
// they should #include <ATen/cuda/CUDAGraphsUtils.cuh>, which has a #pragma once.

// In-kernel call to retrieve philox seed and offset from a  PhiloxCudaState instance whether
// that instance was created with graph capture underway or not.
// See Note [CUDA Graph-safe RNG states].
//
// We can't write a __device__ function in CUDAGeneratorImpl.h, because it's in ATen.
// Also, whatever call unpacks PhiloxCudaState in consumer kernels must be inlineable.
// Easiest thing that comes to mind is, define __device__ helpers here, in ATen/cuda.
//
// The raw definition lives in its own file so jit codegen can easily copy it.
namespace at {
namespace cuda {
namespace philox {

struct SeedOffset {
  __device__ __forceinline__ SeedOffset(uint64_t _seed, uint64_t _offset) : seed_(_seed), offset_(_offset) {}
  __device__ __forceinline__ uint64_t seed() const {
    return seed_;
  }
  __device__ __forceinline__ uint64_t offset() const {
    return offset_;
  }
  private:
  uint64_t seed_;
  uint64_t offset_;
};

__device__ __forceinline__ SeedOffset
unpack(at::PhiloxCudaState arg) {
  if (arg.captured_) {
    // static_cast avoids "warning: invalid narrowing conversion from "long" to "unsigned long".
    return {arg.seed_, static_cast<uint64_t>(*(arg.offset_.ptr)) + arg.offset_intragraph_};
  } else {
    return {arg.seed_, arg.offset_.val};
  }
}

} // namespace philox
} // namespace cuda
} // namespace at
