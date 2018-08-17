#pragma once

/// Defines the Half type (half-precision floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32, instead of using CUDA half intrinisics.
/// Most uses of this type within ATen are memory bound, including the
/// element-wise kernels, and the half intrinisics aren't efficient on all GPUs.
/// If you are writing a compute bound kernel, you can use the CUDA half
/// intrinsics directly on the Half type from device code.

#include <ATen/core/Macros.h>

#include <cmath>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_fp16.h>
#endif

#ifndef AT_HOSTDEVICE
#ifdef __CUDACC__
#define AT_HOSTDEVICE __host__ __device__
#else
#define AT_HOSTDEVICE
#endif
#endif

namespace at {

namespace detail {

AT_CORE_API float halfbits2float(unsigned short bits);
AT_CORE_API unsigned short float2halfbits(float value);

} // namespace detail

struct alignas(2) Half {
  unsigned short x;

  struct from_bits_t {};
  static constexpr from_bits_t from_bits = from_bits_t();

  // HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
  AT_HOSTDEVICE Half() = default;
#else
  Half() = default;
#endif

  constexpr AT_HOSTDEVICE Half(unsigned short bits, from_bits_t) : x(bits){};
  inline AT_HOSTDEVICE Half(float value);
  inline AT_HOSTDEVICE operator float() const;

#ifdef __CUDACC__
  inline AT_HOSTDEVICE Half(const __half& value);
  inline AT_HOSTDEVICE operator __half() const;
#endif
};

template <typename To, typename From>
To convert(From f) {
  return static_cast<To>(f);
}

// skip isnan and isinf check for integral types
template <typename To, typename From>
typename std::enable_if<std::is_integral<From>::value, bool>::type overflows(
    From f) {
  using limit = std::numeric_limits<To>;
  if (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    // allow for negative numbers to wrap using two's complement arithmetic.
    // For example, with uint8, this allows for `a - b` to be treated as
    // `a + 255 * b`.
    return f > limit::max() || (f < 0 && -(uint64_t)f > limit::max());
  } else {
    return f < limit::lowest() || f > limit::max();
  }
}

template <typename To, typename From>
typename std::enable_if<!std::is_integral<From>::value, bool>::type overflows(
    From f) {
  using limit = std::numeric_limits<To>;
  if (limit::has_infinity && std::isinf((double)f)) {
    return false;
  }
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  if (overflows<To, From>(f)) {
    std::ostringstream oss;
    oss << "value cannot be converted to type " << name << " without overflow: " << f;
    throw std::domain_error(oss.str());
  }
  return convert<To, From>(f);
}

template <typename To, typename From>
To HalfFix(From h) {
  To ret;
  ret.x = h.x;
  return ret;
}

AT_CORE_API std::ostream& operator<<(std::ostream& out, const Half& value);

} // namespace at

#include "ATen/core/Half-inl.h"

#undef AT_HOSTDEVICE
