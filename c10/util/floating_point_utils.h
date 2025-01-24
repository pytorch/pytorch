#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/bit_cast.h>
#include <cstdint>

namespace c10::detail {

C10_HOST_DEVICE inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#else
  return c10::bit_cast<float>(w);
#endif
}

C10_HOST_DEVICE inline uint32_t fp32_to_bits(float f) {
#if defined(__OPENCL_VERSION__)
  return as_uint(f);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return (uint32_t)__float_as_uint(f);
#elif defined(__INTEL_COMPILER)
  return _castf32_u32(f);
#else
  return c10::bit_cast<uint32_t>(f);
#endif
}

} // namespace c10::detail
