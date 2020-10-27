#ifndef THC_DEVICE_UTILS_INC
#define THC_DEVICE_UTILS_INC

#include <cuda.h>

#ifdef __HIP_PLATFORM_HCC__
#include <c10/util/Half.h>
#endif

#include <c10/util/BFloat16.h>

/* The largest consecutive integer representable in float32 (2^24) */
#define FLOAT32_MAX_CONSECUTIVE_INT 16777216.0f

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <typename T>
__host__ __device__ __forceinline__ T THCRoundUp(T a, T b) {
  return THCCeilDiv(a, b) * b;
}

/**
 * For CC 3.5+, perform a load using __ldg
 */
template <typename T>
__device__ __forceinline__ T doLdg(const T* p) {
#if __CUDA_ARCH__ >= 350 && !defined __HIP_PLATFORM_HCC__
  return __ldg(p);
#else
  return *p;
#endif
}

#include <ATen/cuda/DeviceUtils.cuh>

#endif // THC_DEVICE_UTILS_INC
