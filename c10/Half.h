#pragma once

/// Defines the Half type (half-precision floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32, instead of using CUDA half intrinisics.
/// Most uses of this type within ATen are memory bound, including the
/// element-wise kernels, and the half intrinisics aren't efficient on all GPUs.
/// If you are writing a compute bound kernel, you can use the CUDA half
/// intrinsics directly on the Half type from device code.

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <iosfwd>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#endif

namespace c10 {

namespace detail {

C10_API float halfbits2float(unsigned short bits);
C10_API unsigned short float2halfbits(float value);

} // namespace detail

struct alignas(2) Half {
  unsigned short x;

  struct from_bits_t {};
  static constexpr from_bits_t from_bits = from_bits_t();

  // HIP wants __host__ __device__ tag, CUDA does not
#ifdef __HIP_PLATFORM_HCC__
  C10_HOST_DEVICE Half() = default;
#else
  Half() = default;
#endif

  constexpr C10_HOST_DEVICE Half(unsigned short bits, from_bits_t) : x(bits){};
  inline C10_HOST_DEVICE Half(float value);
  inline C10_HOST_DEVICE operator float() const;

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_HOST_DEVICE Half(const __half& value);
  inline C10_HOST_DEVICE operator __half() const;
#endif
};

// This is just a placeholder for whatever complex representation we
// end up deciding to use for half-precision complex numbers.
struct alignas(4) ComplexHalf {
  Half real_;
  Half imag_;
  ComplexHalf() = default;
  Half real() const {
    return real_;
  }
  Half imag() const {
    return imag_;
  }
  inline ComplexHalf(std::complex<float> value)
      : real_(value.real()), imag_(value.imag()) {}
  inline operator std::complex<float>() const {
    return {real_, imag_};
  }
};

template <typename T>
struct is_complex_t : public std::false_type {};

template <typename T>
struct is_complex_t<std::complex<T>> : public std::true_type {};

template <>
struct is_complex_t<ComplexHalf> : public std::true_type {};

// Extract double from std::complex<double>; is identity otherwise
// TODO: Write in more idiomatic C++17
template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<std::complex<T>> {
  using type = T;
};
template <>
struct scalar_value_type<ComplexHalf> {
  using type = Half;
};

// The old implementation of Converter as a function made nvcc's head explode
// when we added std::complex on top of the specializations for CUDA-only types
// like __half, so I rewrote it as a templated class (so, no more overloads,
// just (partial) specialization).

template <typename To, typename From, typename Enable = void>
struct Converter {
  To operator()(From f) {
    return static_cast<To>(f);
  }
};

template <typename To, typename From>
To convert(From from) {
  return Converter<To, From>()(from);
}

template <typename To, typename FromV>
struct Converter<
    To,
    std::complex<FromV>,
    typename std::enable_if<
        c10::guts::negation<is_complex_t<To>>::value>::type> {
  To operator()(std::complex<FromV> f) {
    return static_cast<To>(f.real());
  }
};

// skip isnan and isinf check for integral types
template <typename To, typename From>
typename std::enable_if<std::is_integral<From>::value, bool>::type overflows(
    From f) {
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  if (!limit::is_signed && std::numeric_limits<From>::is_signed) {
    // allow for negative numbers to wrap using two's complement arithmetic.
    // For example, with uint8, this allows for `a - b` to be treated as
    // `a + 255 * b`.
    return f > limit::max() ||
        (f < 0 && -static_cast<uint64_t>(f) > limit::max());
  } else {
    return f < limit::lowest() || f > limit::max();
  }
}

template <typename To, typename From>
typename std::enable_if<std::is_floating_point<From>::value, bool>::type
overflows(From f) {
  using limit = std::numeric_limits<typename scalar_value_type<To>::type>;
  if (limit::has_infinity && std::isinf(static_cast<double>(f))) {
    return false;
  }
  if (!limit::has_quiet_NaN && (f != f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

template <typename To, typename From>
typename std::enable_if<is_complex_t<From>::value, bool>::type overflows(
    From f) {
  // casts from complex to real are considered to overflow if the
  // imaginary component is non-zero
  if (!is_complex_t<To>::value && f.imag() != 0) {
    return true;
  }
  // Check for overflow componentwise
  // (Technically, the imag overflow check is guaranteed to be false
  // when !is_complex_t<To>, but any optimizer worth its salt will be
  // able to figure it out.)
  return overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.real()) ||
      overflows<
             typename scalar_value_type<To>::type,
             typename From::value_type>(f.imag());
}

template <typename To, typename From>
To checked_convert(From f, const char* name) {
  if (overflows<To, From>(f)) {
    std::ostringstream oss;
    oss << "value cannot be converted to type " << name
        << " without overflow: " << f;
    throw std::domain_error(oss.str());
  }
  return convert<To, From>(f);
}

C10_API std::ostream& operator<<(std::ostream& out, const Half& value);

} // namespace c10

#include "c10/Half-inl.h"
