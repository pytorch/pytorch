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
#ifndef C10_EMBEDDED
#include <ostream>
#endif // C10_EMBEDDED

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

/*
 * Convert a 16-bit floating-point number in IEEE half-precision format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
C10_HOST_DEVICE inline float fp16_ieee_to_fp32_value(uint16_t h) {
#ifdef C10_X86_F16
  return _cvtsh_ss(h);
#else
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
   * Extract mantissa and biased exponent of the input number into the high bits
   * of the 32-bit word:
   *
   *      +-----+------------+---------------------+
   *      |EEEEE|MM MMMM MMMM|0 0000 0000 0000 0000|
   *      +-----+------------+---------------------+
   * Bits  27-31    17-26            0-16
   */
  const uint32_t two_w = w + w;

  /*
   * Shift mantissa and exponent into bits 23-28 and bits 13-22 so they become
   * mantissa and exponent of a single-precision floating-point number:
   *
   *       S|Exponent |          Mantissa
   *      +-+---+-----+------------+----------------+
   *      |0|000|EEEEE|MM MMMM MMMM|0 0000 0000 0000|
   *      +-+---+-----+------------+----------------+
   * Bits   | 23-31   |           0-22
   *
   * Next, there are some adjustments to the exponent:
   * - The exponent needs to be corrected by the difference in exponent bias
   * between single-precision and half-precision formats (0x7F - 0xF = 0x70)
   * - Inf and NaN values in the inputs should become Inf and NaN values after
   * conversion to the single-precision number. Therefore, if the biased
   * exponent of the half-precision input was 0x1F (max possible value), the
   * biased exponent of the single-precision output must be 0xFF (max possible
   * value). We do this correction in two steps:
   *   - First, we adjust the exponent by (0xFF - 0x1F) = 0xE0 (see exp_offset
   * below) rather than by 0x70 suggested by the difference in the exponent bias
   * (see above).
   *   - Then we multiply the single-precision result of exponent adjustment by
   * 2**(-112) to reverse the effect of exponent adjustment by 0xE0 less the
   * necessary exponent adjustment by 0x70 due to difference in exponent bias.
   *     The floating-point multiplication hardware would ensure than Inf and
   * NaN would retain their value on at least partially IEEE754-compliant
   * implementations.
   *
   * Note that the above operations do not handle denormal inputs (where biased
   * exponent == 0). However, they also do not operate on denormal inputs, and
   * do not produce denormal results.
   */
  constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
  // const float exp_scale = 0x1.0p-112f;
  constexpr uint32_t scale_bits = (uint32_t)15 << 23;
  float exp_scale_val = 0;
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  const float exp_scale = exp_scale_val;
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  /*
   * Convert denormalized half-precision inputs into single-precision results
   * (always normalized). Zero inputs are also handled here.
   *
   * In a denormalized number the biased exponent is zero, and mantissa has
   * on-zero bits. First, we shift mantissa into bits 0-9 of the 32-bit word.
   *
   *                  zeros           |  mantissa
   *      +---------------------------+------------+
   *      |0000 0000 0000 0000 0000 00|MM MMMM MMMM|
   *      +---------------------------+------------+
   * Bits             10-31                0-9
   *
   * Now, remember that denormalized half-precision numbers are represented as:
   *    FP16 = mantissa * 2**(-24).
   * The trick is to construct a normalized single-precision number with the
   * same mantissa and thehalf-precision input and with an exponent which would
   * scale the corresponding mantissa bits to 2**(-24). A normalized
   * single-precision floating-point number is represented as: FP32 = (1 +
   * mantissa * 2**(-23)) * 2**(exponent - 127) Therefore, when the biased
   * exponent is 126, a unit change in the mantissa of the input denormalized
   * half-precision number causes a change of the constructed single-precision
   * number by 2**(-24), i.e. the same amount.
   *
   * The last step is to adjust the bias of the constructed single-precision
   * number. When the input half-precision number is zero, the constructed
   * single-precision number has the value of FP32 = 1 * 2**(126 - 127) =
   * 2**(-1) = 0.5 Therefore, we need to subtract 0.5 from the constructed
   * single-precision number to get the numerical equivalent of the input
   * half-precision number.
   */
  constexpr uint32_t magic_mask = UINT32_C(126) << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  /*
   * - Choose either results of conversion of input as a normalized number, or
   * as a denormalized number, depending on the input exponent. The variable
   * two_w contains input exponent in bits 27-31, therefore if its smaller than
   * 2**27, the input is either a denormal number, or zero.
   * - Combine the result of conversion of exponent and mantissa with the sign
   * of the input number.
   */
  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
      (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                   : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
#endif // C10_X86_F16
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 16-bit floating-point number in IEEE half-precision format, in bit
 * representation.
 *
 * @note The implementation relies on IEEE-like (no assumption about rounding
 * mode and no operations on denormals) floating-point operations and bitcasts
 * between integer and floating-point variables.
 */
inline uint16_t fp16_ieee_from_fp32_value(float f) {
#ifdef C10_X86_F16
  return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
  // const float scale_to_inf = 0x1.0p+112f;
  // const float scale_to_zero = 0x1.0p-110f;
  constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
  float scale_to_inf_val = 0, scale_to_zero_val = 0;
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(
      &scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

#if defined(_MSC_VER) && _MSC_VER == 1916
  float base = ((signbit(f) != 0 ? -f : f) * scale_to_inf) * scale_to_zero;
#else
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
#endif

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
#endif // C10_X86_F16
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

#ifndef C10_EMBEDDED
C10_API inline std::ostream& operator<<(std::ostream& out, const Half& value) {
  out << (float)value;
  return out;
}
#endif // C10_EMBEDDED

} // namespace c10

#include <c10/util/Half-inl.h> // IWYU pragma: keep
