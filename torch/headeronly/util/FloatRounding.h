#pragma once

// Narrowing floating-point conversions with an explicit IEEE 754 rounding
// direction, implemented with integer arithmetic so that they are portable
// (host and device) and cover non-IEEE formats such as float8.
//
// Semantics:
// - The exact source value is rounded once to the destination format.
// - On overflow, results rounded toward zero become the largest finite value;
//   otherwise they become infinity or, for formats without infinity, the value
//   that the default conversion produces on overflow (saturation for
//   Float8_e4m3fn and Float8_e5m3fnu, NaN for the fnuz formats and
//   Float8_e8m0fnu).
// - For the unsigned formats (Float8_e8m0fnu and Float8_e5m3fnu), the
//   magnitude of the source is rounded.
// - Float8_e8m0fnu has no zero: values below the smallest power of two clamp
//   to it, and ties of ToNearestEven round up (it has no mantissa bits to be
//   even).

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Float8_e5m3fnu.h>
#include <torch/headeronly/util/Float8_e8m0fnu.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/floating_point_utils.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#if __has_include(<version>)
#include <version>
#endif
#ifdef __cpp_lib_bit_cast
#include <bit>
#endif
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace c10 {

enum class RoundingMode : uint8_t {
  ToNearestEven,
  TowardZero,
  TowardPositive,
  TowardNegative,
};

namespace detail {

// Unsigned integer with the size of T, to copy the bits of T to and from.
template <typename T>
using Bits = std::conditional_t<
    sizeof(T) == 1,
    uint8_t,
    std::conditional_t<
        sizeof(T) == 2,
        uint16_t,
        std::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;

// Whether the default conversion to T saturates to the largest finite value on
// overflow, rather than producing infinity or, without infinity, NaN.
template <typename T>
struct Saturates : std::false_type {};
template <>
struct Saturates<Float8_e4m3fn> : std::true_type {};
template <>
struct Saturates<Float8_e5m3fnu> : std::true_type {};

// memcpy rather than std::bit_cast, which needs C++20 and, in device code,
// nvcc's --expt-relaxed-constexpr.
template <typename To, typename From>
C10_HOST_DEVICE inline To bit_copy(const From& from) {
  static_assert(sizeof(To) == sizeof(From), "bit_copy needs equal sizes");
  To to;
  std::memcpy(&to, &from, sizeof(To));
  return to;
}

// The result of rounding past the largest finite magnitude of T.
template <typename T>
constexpr T overflow_value() {
  using L = std::numeric_limits<T>;
  if constexpr (Saturates<T>::value) {
    return L::max();
  } else if constexpr (L::has_infinity) {
    return L::infinity();
  } else {
    return L::quiet_NaN();
  }
}

// Rounds x to float, setting the last bit if inexact ("round to odd"). float
// has at least two more mantissa bits than the narrower formats and covers
// their ranges, so rounding the result to one of them again rounds x correctly.
C10_HOST_DEVICE inline float round_to_odd_float(double x) {
  const auto f = static_cast<float>(x);
  const auto back = static_cast<double>(f);
  auto bits = fp32_to_bits(f);
  // A round-to-nearest result that is inexact (false for NaN) with an even
  // last bit is stepped one ulp toward x, without branches.
  const uint32_t step = (back != x) & (x == x) & ~bits & 1u;
  const uint32_t up = (back < 0 ? -back : back) < (x < 0 ? -x : x);
  bits += step * (2 * up - 1);
  return fp32_from_bits(bits);
}

// Requires x != 0. This is std::countl_zero, which needs C++20 and, in device
// code, nvcc's --expt-relaxed-constexpr.
C10_HOST_DEVICE inline int leading_zeros(uint64_t x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __clzll(static_cast<long long>(x));
#elif defined(_MSC_VER) && !defined(__clang__) && !defined(__SYCL_DEVICE_ONLY__)
  unsigned long index = 0;
  _BitScanReverse64(&index, x);
  return 63 - static_cast<int>(index);
#else
  return __builtin_clzll(x);
#endif
}

} // namespace detail

// Converts a float or double to a narrower floating-point type with the given
// rounding direction. Half and BFloat16 sources should be converted to float
// first, which is exact.
//
// Src and Dst are described by their std::numeric_limits. Dst's finite
// magnitudes, in increasing order, must have consecutive encodings starting at
// 0, which holds for IEEE-like formats including their subnormals.
template <typename Dst, typename Src>
C10_HOST_DEVICE Dst convert_with_rounding(Src x, RoundingMode mode) {
  using detail::Bits;
  using SrcL = std::numeric_limits<Src>;
  using DstL = std::numeric_limits<Dst>;
  static_assert(SrcL::is_iec559, "the source must be an IEEE 754 format");
  static_assert(
      DstL::is_specialized && !DstL::is_integer && DstL::radix == 2,
      "the destination must be a binary floating-point format");

  constexpr int kSrcMantBits = SrcL::digits - 1;
  // The bits that are neither the sign nor the mantissa.
  constexpr int kSrcExpBits = sizeof(Src) * 8 - SrcL::digits;
  constexpr uint32_t kSrcExpMax = (1u << kSrcExpBits) - 1;
  // Exponent of the smallest normal value.
  constexpr int kSrcExpMin = SrcL::min_exponent - 1;
  constexpr int kSrcBias = 1 - kSrcExpMin;

  constexpr int kDstMantBits = DstL::digits - 1;
  constexpr bool kDstHasSubnormals = DstL::has_denorm == std::denorm_present;
  constexpr int kDstExpMin = DstL::min_exponent - 1;
  constexpr int kDstBias = (kDstHasSubnormals ? 1 : 0) - kDstExpMin;
  constexpr uint32_t kDstSignBit =
      DstL::is_signed ? uint32_t{1} << (sizeof(Dst) * 8 - 1) : 0;
  // Values are constexpr (numeric_limits must not be called at run time in
  // device code); their bits are copied at run time.
  constexpr Dst kDstMaxValue = DstL::max();
  constexpr Dst kDstNaNValue = DstL::quiet_NaN();
  constexpr Dst kDstOverflowValue = detail::overflow_value<Dst>();
  const uint32_t kDstMaxFinite = detail::bit_copy<Bits<Dst>>(kDstMaxValue);
  const uint32_t kDstNaN = detail::bit_copy<Bits<Dst>>(kDstNaNValue);
  const uint32_t kDstOverflow = detail::bit_copy<Bits<Dst>>(kDstOverflowValue);
  // The fnuz formats encode NaN as negative zero.
  const bool kDstHasNegZero = DstL::is_signed && kDstNaN != kDstSignBit;

  const auto from_bits = [](uint32_t bits) {
    return detail::bit_copy<Dst>(static_cast<Bits<Dst>>(bits));
  };

  auto src_enc = detail::bit_copy<Bits<Src>>(x);
  auto src_exp = static_cast<uint32_t>((src_enc >> kSrcMantBits) & kSrcExpMax);
  uint64_t src_mant = src_enc & ((Bits<Src>{1} << kSrcMantBits) - 1);
  bool neg = DstL::is_signed && (src_enc >> (sizeof(Src) * 8 - 1)) != 0;
  // Sign bit to apply to a nonzero result; the fnuz NaN is 0x80 either way.
  uint32_t dst_sign = neg ? kDstSignBit : 0;

  if (src_exp == kSrcExpMax) {
    return from_bits((src_mant != 0 ? kDstNaN : kDstOverflow) | dst_sign);
  }
  if (src_exp == 0 && src_mant == 0) {
    return from_bits(kDstHasNegZero ? dst_sign : 0);
  }

  bool src_normal = src_exp > 0;

  // |x| = explicit_mantissa * 2^(unbiased_exponent - kSrcMantBits), where
  // explicit_mantissa has its leading 1 explicitly at bit kSrcMantBits.
  uint64_t explicit_mantissa = 0;
  int unbiased_exponent = 0;
  if (src_normal) {
    // Make the implicit leading 1 explicit.
    explicit_mantissa = src_mant | (uint64_t{1} << kSrcMantBits);
    unbiased_exponent = static_cast<int>(src_exp) - kSrcBias;
  } else {
    // Normalize the subnormal, which takes unbiased_exponent below kSrcExpMin.
    int norm_shift = detail::leading_zeros(src_mant) - (63 - kSrcMantBits);
    explicit_mantissa = src_mant << norm_shift;
    unbiased_exponent = kSrcExpMin - norm_shift;
  }

  bool dst_normal = unbiased_exponent >= kDstExpMin;

  if (!dst_normal && !kDstHasSubnormals) {
#ifdef __cpp_lib_bit_cast
    static_assert(
        kDstHasSubnormals || std::bit_cast<Bits<Dst>>(DstL::min()) == 0,
        "a destination without subnormals must have no zero either, so that "
        "encoding 0 is its smallest normal value");
#endif
    return from_bits(0);
  }

  // Number of low bits of explicit_mantissa below Dst's quantum at this
  // exponent.
  static_assert(
      kSrcMantBits > kDstMantBits,
      "the destination must have fewer mantissa bits than the source");
  int shift = kSrcMantBits - kDstMantBits;
  // For subnormals we allow the leading 1 to move to a lower position.
  if (!dst_normal) {
    shift += kDstExpMin - unbiased_exponent;
  }
  // Shifts of 64 or more are undefined behavior, hence clamp. This doesn't
  // change the result: beyond kSrcMantBits + 1 even the leading 1 is below the
  // guard bit, so the value only sets sticky.
  shift = shift < kSrcMantBits + 2 ? shift : kSrcMantBits + 2;
  // The bits from the quantum up are kept, the one below is the guard bit and
  // the rest are sticky.
  uint64_t kept = explicit_mantissa >> shift;
  bool guard = (explicit_mantissa >> (shift - 1)) & 1;
  bool sticky = (explicit_mantissa & ((uint64_t{1} << (shift - 1)) - 1)) != 0;

  // Dst's unsigned encoding of the value truncated to Dst's precision.
  uint64_t dst_exp = 0;
  uint64_t dst_mant = 0;
  if (dst_normal) {
    dst_exp = static_cast<uint64_t>(unbiased_exponent + kDstBias);
    dst_mant = kept - (uint64_t{1} << kDstMantBits); // drop the leading 1
  } else {
    dst_mant = kept;
  }

  uint64_t dst_enc = (dst_exp << kDstMantBits) | dst_mant;

  // Consecutive magnitudes have consecutive encodings, so rounding up is a +1
  // that carries into the exponent, or past the largest finite value, by
  // itself.
  bool to_zero = mode == RoundingMode::TowardZero ||
      (mode == RoundingMode::TowardPositive && neg) ||
      (mode == RoundingMode::TowardNegative && !neg);
  if (mode == RoundingMode::ToNearestEven) {
    dst_enc += guard && (sticky || (dst_enc & 1) || kDstMantBits == 0);
  } else if (!to_zero) {
    dst_enc += guard || sticky;
  }

  if (dst_enc > kDstMaxFinite) {
    return from_bits((to_zero ? kDstMaxFinite : kDstOverflow) | dst_sign);
  }
  if (dst_enc == 0 && !kDstHasNegZero) {
    return from_bits(0);
  }
  return from_bits(static_cast<uint32_t>(dst_enc) | dst_sign);
}

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::convert_with_rounding;
using c10::RoundingMode;
HIDDEN_NAMESPACE_END(torch, headeronly)
