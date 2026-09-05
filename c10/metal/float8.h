#pragma once
#include <metal_stdlib>

// Bit-exact Metal ports of the fp8 <-> fp32 conversions from
// torch/headeronly/util/Float8_*.h, which Metal sources cannot include.
// Keep in sync with those files; see them for the bit-level derivations
// of the constants below.

namespace c10 {
namespace metal {
namespace detail {

inline float fp8e4m3fn_to_fp32(uchar input) {
  const uint w = uint(input) << 24;
  const uint sign = w & 0x80000000;
  const uint nonsign = w & 0x7fffffff;
  uint renorm_shift = nonsign == 0 ? 0 : ::metal::clz(nonsign);
  renorm_shift = renorm_shift > 4 ? renorm_shift - 4 : 0;
  const uint nan_mask = nonsign == 0x7f000000 ? 0x7f800000 : 0;
  const uint zero_mask = nonsign == 0 ? 0xffffffff : 0;
  const uint result = sign |
      ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
        nan_mask) &
       ~zero_mask);
  return as_type<float>(result);
}

inline uchar fp8e4m3fn_from_fp32(float value) {
  constexpr uint fp8_max = 1087u << 20;
  constexpr uint denorm_mask = 141u << 23;
  uint bits = as_type<uint>(value);
  const uint sign = bits & 0x80000000;
  bits ^= sign;
  uchar result;
  if (bits >= fp8_max) {
    result = bits > 0x7f800000 ? 0x7f : 0x7e;
  } else if (bits < (121u << 23)) {
    bits = as_type<uint>(as_type<float>(bits) + as_type<float>(denorm_mask));
    result = static_cast<uchar>(bits - denorm_mask);
  } else {
    const uchar mantissa_odd = (bits >> 20) & 1;
    bits += 0xc4000000 + 0x7ffff + mantissa_odd;
    result = static_cast<uchar>(bits >> 20);
    result = result == 0x7f ? 0x7e : result;
  }
  return result | static_cast<uchar>(sign >> 24);
}

inline float fp8e5m2_to_fp32(uchar input) {
  const ushort h_bits = ushort(input) << 8;
  // The hardware half->float conversion canonicalizes NaNs; expand NaN
  // manually to match the CPU conversion bit-for-bit (sign and mantissa
  // payload preserved, quiet bit set).
  if ((h_bits & 0x7fff) > 0x7c00) {
    const uint sign = (uint(h_bits) & 0x8000) << 16;
    const uint payload = (uint(h_bits) & 0x03ff) << 13;
    return as_type<float>(sign | 0x7fc00000 | payload);
  }
  return float(as_type<half>(h_bits));
}

inline uchar fp8e5m2_from_fp32(float value) {
  constexpr uint fp8_max = 143u << 23;
  constexpr uint denorm_mask = 134u << 23;
  uint bits = as_type<uint>(value);
  const uint sign = bits & 0x80000000;
  bits ^= sign;
  uchar result;
  if (bits >= fp8_max) {
    result = bits > 0x7f800000 ? 0x7f : 0x7c;
  } else if (bits < (113u << 23)) {
    bits = as_type<uint>(as_type<float>(bits) + as_type<float>(denorm_mask));
    result = static_cast<uchar>(bits - denorm_mask);
  } else {
    const uint mantissa_odd = (bits >> 21) & 1;
    bits += 0xc8000000 + 0xfffff + mantissa_odd;
    result = static_cast<uchar>(bits >> 21);
  }
  return result | static_cast<uchar>(sign >> 24);
}

inline float fp8e8m0fnu_to_fp32(uchar input) {
  if (input == 0) {
    return as_type<float>(0x00400000u);
  }
  if (input == 0xff) {
    return as_type<float>(0x7f800001u);
  }
  return as_type<float>(uint(input) << 23);
}

inline uchar fp8e8m0fnu_from_fp32(float value) {
  const uint bits = as_type<uint>(value);
  uint exponent = (bits >> 23) & 0xff;
  if (exponent == 0xff) {
    return 0xff;
  }
  const bool guard = bits & 0x400000;
  const bool round = bits & 0x200000;
  const bool sticky = bits & 0x1fffff;
  exponent += guard && (round || sticky || exponent > 0);
  return static_cast<uchar>(exponent);
}

} // namespace detail

struct alignas(1) float8_e4m3fn {
  uchar x;
  float8_e4m3fn() = default;
  template <typename T>
  float8_e4m3fn(T value) : x(detail::fp8e4m3fn_from_fp32(float(value))) {}
  operator float() const {
    return detail::fp8e4m3fn_to_fp32(x);
  }
};

struct alignas(1) float8_e5m2 {
  uchar x;
  float8_e5m2() = default;
  template <typename T>
  float8_e5m2(T value) : x(detail::fp8e5m2_from_fp32(float(value))) {}
  operator float() const {
    return detail::fp8e5m2_to_fp32(x);
  }
};

struct alignas(1) float8_e8m0fnu {
  uchar x;
  float8_e8m0fnu() = default;
  template <typename T>
  float8_e8m0fnu(T value) : x(detail::fp8e8m0fnu_from_fp32(float(value))) {}
  operator float() const {
    return detail::fp8e8m0fnu_to_fp32(x);
  }
};

template <typename T>
constexpr constant bool is_float8_v = ::metal::is_same_v<T, float8_e4m3fn> ||
    ::metal::is_same_v<T, float8_e5m2> || ::metal::is_same_v<T, float8_e8m0fnu>;

} // namespace metal
} // namespace c10
