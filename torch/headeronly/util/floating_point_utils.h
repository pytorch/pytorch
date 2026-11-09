#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/bit_cast.h>
#include <cstdint>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly, detail)

C10_HOST_DEVICE inline float fp32_from_bits(uint32_t w) {
#if defined(__OPENCL_VERSION__)
  return as_float(w);
#elif defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float((unsigned int)w);
#elif defined(__INTEL_COMPILER)
  return _castu32_f32(w);
#else
  return torch::headeronly::bit_cast<float>(w);
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
  return torch::headeronly::bit_cast<uint32_t>(f);
#endif
}

// Rounds x to float, setting the last bit if inexact ("round to odd"). float
// has at least two more mantissa bits than the narrower formats and covers
// their ranges, so rounding the result to one of them again rounds x correctly.
C10_HOST_DEVICE inline float round_to_odd_float(double x) {
#if defined(__CUDA_ARCH__)
  // Truncate, then set the last bit if inexact (false for NaN). Not used for
  // HIP, where it is untested; the portable version below is correct there.
  const float t = __double2float_rz(x);
  return fp32_from_bits(
      fp32_to_bits(t) | ((static_cast<double>(t) != x) & (x == x)));
#else
  const auto f = static_cast<float>(x);
  const auto back = static_cast<double>(f);
  auto bits = fp32_to_bits(f);
  // A round-to-nearest result that is inexact (false for NaN) with an even
  // last bit is stepped one ulp toward x, without branches.
  const uint32_t step = (back != x) & (x == x) & ~bits & 1u;
  const uint32_t up = (back < 0 ? -back : back) < (x < 0 ? -x : x);
  bits += step * (2 * up - 1);
  return fp32_from_bits(bits);
#endif
}

HIDDEN_NAMESPACE_END(torch, headeronly, detail)

namespace c10::detail {
using torch::headeronly::detail::fp32_from_bits;
using torch::headeronly::detail::fp32_to_bits;
using torch::headeronly::detail::round_to_odd_float;
} // namespace c10::detail
