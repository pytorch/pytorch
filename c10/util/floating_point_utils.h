#pragma once

#include <cstdint>

// Standard check for compiling CUDA with clang
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
#define C10_DEVICE_HOST_FUNCTION __device__ __host__
#else
#define C10_DEVICE_HOST_FUNCTION
#endif

namespace c10 {

namespace detail {

C10_DEVICE_HOST_FUNCTION inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#else
  union {
    uint32_t as_bits;
    float as_value;
  } fp32 = {w};
  return fp32.as_value;
#endif
}

C10_DEVICE_HOST_FUNCTION inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#else
  union {
    float as_value;
    uint32_t as_bits;
  } fp32 = {f};
  return fp32.as_bits;
#endif
}

} // namespace detail
} // namespace c10
