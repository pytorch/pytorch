#pragma once

#include <c10/metal/reduction_utils.h>

template <typename T>
struct svd_real {
  using type = T;
};
template <>
struct svd_real<float2> {
  using type = float;
};
template <typename T>
using svd_real_t = typename svd_real<T>::type;

inline float svd_abs2(float z) {
  return z * z;
}
inline float svd_abs2(float2 z) {
  return z.x * z.x + z.y * z.y;
}
inline float svd_conjmul(float a, float b) {
  return a * b;
}
inline float2 svd_conjmul(float2 a, float2 b) {
  return float2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}
inline float svd_conj(float z) {
  return z;
}
inline float2 svd_conj(float2 z) {
  return float2(z.x, -z.y);
}
inline float svd_mul(float a, float b) {
  return a * b;
}
inline float2 svd_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float svd_simd_sum(float v) {
  return c10::metal::simd_sum(v);
}
inline float2 svd_simd_sum(float2 v) {
  return float2(c10::metal::simd_sum(v.x), c10::metal::simd_sum(v.y));
}
inline float svd_one(float) {
  return 1.0f;
}
inline float2 svd_one(float2) {
  return float2(1.0f, 0.0f);
}
inline float svd_real_part(float z) {
  return z;
}
inline float svd_real_part(float2 z) {
  return z.x;
}
// NB: float2(x) -> (x,x), so build real T explicitly.
inline float svd_from_real(float, float x) {
  return x;
}
inline float2 svd_from_real(float2, float x) {
  return float2(x, 0.0f);
}
