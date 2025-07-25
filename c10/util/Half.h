#pragma once

/// Defines the Half type (half-precision floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32, instead of using CUDA half intrinsics.
/// Most uses of this type within ATen are memory bound, including the
/// element-wise kernels, and the half intrinsics aren't efficient on all GPUs.
/// If you are writing a compute bound kernel, you can use the CUDA half
/// intrinsics directly on the Half type from device code.

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/bit_cast.h>
#include <c10/util/floating_point_utils.h>
#include <torch/headeronly/util/Half.h>
#include <type_traits>

#if defined(__cplusplus)
#include <cmath>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <ostream>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#ifdef __HIPCC__
#include <hip/hip_fp16.h>
#endif

#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp> // for SYCL 1.2.1
#elif defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp> // for SYCL 2020
#endif

#if defined(__aarch64__) && !defined(__CUDACC__)
#include <arm_neon.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
#if defined(__F16C__) &&                               \
    !(defined(__CUDA_ARCH__) || defined(__CUDACC__) || \
      defined(__HIP_DEVICE_COMPILE__))
#define C10_X86_F16 1
#include <immintrin.h> // import conversion ops from f16cintrin.h
#endif // defined(__F16C__) && !(defined(__CUDA_ARCH__) || defined(__CUDACC__)
       // || defined(__HIP_DEVICE_COMPILE__))
#endif // __x86_64__ || _M_X64 || __i386 || _M_IX86
#endif // __GNUC__ || __clang__

namespace c10 {

namespace detail {

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
  /*
   * Extend the half-precision floating-point number to 32 bits and shift to the
   * upper part of the 32-bit word:
   *      +---+-----+------------+-------------------+
   *      | S |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  31  26-30    16-25            0-15
   *
   * S - sign bit, E - bits of the biased exponent, M - bits of the mantissa, 0
   * - zero bits.
   */
  const uint32_t w = (uint32_t)h << 16;
  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = w & UINT32_C(0x80000000);
  /*
   * Extract mantissa and biased exponent of the input number into the bits 0-30
   * of the 32-bit word:
   *
   *      +---+-----+------------+-------------------+
   *      | 0 |EEEEE|MM MMMM MMMM|0000 0000 0000 0000|
   *      +---+-----+------------+-------------------+
   * Bits  30  27-31     17-26            0-16
   */
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  /*
   * Renorm shift is the number of bits to shift mantissa left to make the
   * half-precision number normalized. If the initial number is normalized, some
   * of its high 6 bits (sign == 0 and 5-bit exponent) equals one. In this case
   * renorm_shift == 0. If the number is denormalize, renorm_shift > 0. Note
   * that if we shift denormalized nonsign by renorm_shift, the unit bit of
   * mantissa will shift into exponent, turning the biased exponent into 1, and
   * making mantissa normalized (i.e. without leading 1).
   */
#ifdef _MSC_VER
  unsigned long nonsign_bsr;
  _BitScanReverse(&nonsign_bsr, (unsigned long)nonsign);
  uint32_t renorm_shift = (uint32_t)nonsign_bsr ^ 31;
#else
  uint32_t renorm_shift = __builtin_clz(nonsign);
#endif
  renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
  /*
   * Iff half-precision number has exponent of 15, the addition overflows
   * it into bit 31, and the subsequent shift turns the high 9 bits
   * into 1. Thus inf_nan_mask == 0x7F800000 if the half-precision number
   * had exponent of 15 (i.e. was NaN or infinity) 0x00000000 otherwise
   */
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  /*
   * Iff nonsign is 0, it overflows into 0xFFFFFFFF, turning bit 31
   * into 1. Otherwise, bit 31 remains 0. The signed shift right by 31
   * broadcasts bit 31 into all bits of the zero_mask. Thus zero_mask ==
   * 0xFFFFFFFF if the half-precision number was zero (+0.0h or -0.0h)
   * 0x00000000 otherwise
   */
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  /*
   * 1. Shift nonsign left by renorm_shift to normalize it (if the input
   * was denormal)
   * 2. Shift nonsign right by 3 so the exponent (5 bits originally)
   * becomes an 8-bit field and 10-bit mantissa shifts into the 10 high
   * bits of the 23-bit mantissa of IEEE single-precision number.
   * 3. Add 0x70 to the exponent (starting at bit 23) to compensate the
   * different in exponent bias (0x7F for single-precision number less 0xF
   * for half-precision number).
   * 4. Subtract renorm_shift from the exponent (starting at bit 23) to
   * account for renormalization. As renorm_shift is less than 0x70, this
   * can be combined with step 3.
   * 5. Binary OR with inf_nan_mask to turn the exponent into 0xFF if the
   * input was NaN or infinity.
   * 6. Binary ANDNOT with zero_mask to turn the mantissa and exponent
   * into zero if the input was zero.
   * 7. Combine with the sign of the input number.
   */
  return sign |
      ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
        inf_nan_mask) &
       ~zero_mask);
}

#ifdef C10_X86_F16
#undef C10_X86_F16
#endif // C10_X86_F16

#if defined(__aarch64__) && !defined(__CUDACC__)
inline float16_t fp16_from_bits(uint16_t h) {
  return c10::bit_cast<float16_t>(h);
}

inline uint16_t fp16_to_bits(float16_t f) {
  return c10::bit_cast<uint16_t>(f);
}

// According to https://godbolt.org/z/frExdbsWG it would translate to single
// fcvt s0, h0
inline float native_fp16_to_fp32_value(uint16_t h) {
  return static_cast<float>(fp16_from_bits(h));
}

inline uint16_t native_fp16_from_fp32_value(float f) {
  return fp16_to_bits(static_cast<float16_t>(f));
}
#endif

} // namespace detail

struct alignas(2) Half {
  unsigned short x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  // HIP wants __host__ __device__ tag, CUDA does not
#if defined(USE_ROCM)
  C10_HOST_DEVICE Half() = default;
#else
  Half() = default;
#endif

  constexpr C10_HOST_DEVICE Half(unsigned short bits, from_bits_t) : x(bits) {}
#if defined(__aarch64__) && !defined(__CUDACC__)
  inline Half(float16_t value);
  inline operator float16_t() const;
#else
  inline C10_HOST_DEVICE Half(float value);
  inline C10_HOST_DEVICE operator float() const;
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
  inline C10_HOST_DEVICE Half(const __half& value);
  inline C10_HOST_DEVICE operator __half() const;
#endif
#ifdef SYCL_LANGUAGE_VERSION
  inline C10_HOST_DEVICE Half(const sycl::half& value);
  inline C10_HOST_DEVICE operator sycl::half() const;
#endif
};

inline std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << (float)value;
  return out;
}

} // namespace c10

#include <c10/util/Half-inl.h> // IWYU pragma: keep
