#pragma once

#include <torch/headeronly/util/floating_point_utils.h>

#include <cstdint>

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

namespace torch::headeronly::detail {

/*
 * Convert a 8-bit floating-point number in either f8 E4M3FNUZ or bf8 E5M2FNUZ
 * format, in bit representation, to a 32-bit floating-point number.
 */
template <uint32_t we, uint32_t wm>
inline C10_HOST_DEVICE float fp8_fnuz_to_fp32_value(uint8_t x) {
  static_assert((we == 4 && wm == 3) || (we == 5 && wm == 2));
  constexpr uint32_t weo = 8;
  constexpr uint32_t wmo = 23;

  if (x == 0) {
    return 0;
  }

  if (x == 0x80) {
    constexpr uint32_t ifNaN = 0x7F800001;
    return fp32_from_bits(ifNaN);
  }

  uint32_t mantissa = x & ((1 << wm) - 1);
  uint32_t exponent = (x & 0x7F) >> wm;

  // subnormal input
  if (exponent == 0) {
    // guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    uint32_t renorm_shift = __clz(mantissa);
#elif defined(__SYCL_DEVICE_ONLY__)
    uint32_t renorm_shift = sycl::clz(mantissa);
#elif defined(_MSC_VER)
    unsigned long nonsign_bsr;
    _BitScanReverse(&nonsign_bsr, (unsigned long)mantissa);
    uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
    uint32_t renorm_shift = __builtin_clz(mantissa);
#endif
    uint32_t sh = 1 + renorm_shift - (32 - wm);
    mantissa <<= sh;
    exponent += 1 - sh;
    mantissa &= ((1 << wm) - 1);
  }

  const uint32_t exp_low_cutoff = (1 << (weo - 1)) - (1 << (we - 1));
  exponent += exp_low_cutoff - 1;
  mantissa <<= wmo - wm;

  uint32_t sign = x >> 7;
  uint32_t retval = (sign << 31) | (exponent << 23) | mantissa;
  return fp32_from_bits(retval);
}

} // namespace torch::headeronly::detail

namespace c10::detail {
using torch::headeronly::detail::fp8_fnuz_to_fp32_value;
}
