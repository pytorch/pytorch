#include "THCHalf.h"
#include "THCThrustAllocator.cuh"
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

struct __half2floatOp {
  __device__ float operator()(half v) { return __half2float(v); }
};

struct __float2halfOp {
  __device__ half operator()(float v) { return __float2half(v); }
};

void THCFloat2Half(THCState *state, half *out, float *in, ptrdiff_t len) {
  THCThrustAllocator thrustAlloc(state);
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __float2halfOp());
}

void THCHalf2Float(THCState *state, float *out, half *in, ptrdiff_t len) {
  THCThrustAllocator thrustAlloc(state);
  thrust::transform(
#if CUDA_VERSION >= 7000
    thrust::cuda::par(thrustAlloc).on(THCState_getCurrentStream(state)),
#else
    thrust::device,
#endif
    in, in + len, out, __half2floatOp());
}

THC_EXTERNC int THC_nativeHalfInstructions(THCState *state) {
  cudaDeviceProp* prop =
    THCState_getCurrentDeviceProperties(state);

  // CC 5.3+
  return (prop->major > 5 ||
          (prop->major == 5 && prop->minor == 3));
}

THC_EXTERNC int THC_fastHalfInstructions(THCState *state) {
  cudaDeviceProp* prop =
    THCState_getCurrentDeviceProperties(state);

  // Check for CC 6.0 only (corresponds to P100)
  return (prop->major == 6 && prop->minor == 0);
}
