#pragma once
#include <metal_stdlib>

// Metal ports of the fp8 <-> fp32 conversions from
// torch/headeronly/util/Float8_*.h. Those headers cannot be included as-is
// in Metal shaders: they depend on the C++ standard library and use double.

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

template <typename T>
constexpr constant bool is_float8_v = ::metal::is_same_v<T, float8_e4m3fn>;

} // namespace metal
} // namespace c10
