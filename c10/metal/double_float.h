// Double-float arithmetic, for kernels that need more precision than float32
#pragma once
#include <metal_stdlib>

namespace c10 {
namespace metal {

// A value held as the unevaluated sum hi + lo of two float32s, with
// |lo| <= ulp(hi) / 2, carrying roughly 48 bits of significand. Metal Shading
// Language has no `double`, so this is how a kernel keeps an intermediate at
// better than float32 precision.
//
// The transformations below are exact under IEEE semantics and only under
// those: they depend on the compiler not reassociating the expressions, which
// holds because the shaders are built with -fno-fast-math (cmake/Metal.cmake)
// and compiled at runtime with MTLMathModeSafe unless PYTORCH_MPS_FAST_MATH
// asks otherwise.
struct df32 {
  float hi;
  float lo;
};

// Knuth's two-sum: the result is a + b exactly, for any a and b.
inline df32 two_sum(float a, float b) {
  const float s = a + b;
  const float b_virtual = s - a;
  return df32{s, (a - (s - b_virtual)) + (b - b_virtual)};
}

// Dekker's fast two-sum. Same result as two_sum, but only when |a| >= |b|.
inline df32 quick_two_sum(float a, float b) {
  const float s = a + b;
  return df32{s, b - (s - a)};
}

// Exact product: the residual is whatever the multiply rounded away, which
// fma() recovers.
inline df32 two_prod(float a, float b) {
  const float p = a * b;
  return df32{p, ::metal::fma(a, b, -p)};
}

inline df32 df_add(df32 a, df32 b) {
  df32 s = two_sum(a.hi, b.hi);
  s.lo += a.lo + b.lo;
  return quick_two_sum(s.hi, s.lo);
}

inline df32 df_mul(df32 a, df32 b) {
  df32 p = two_prod(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return quick_two_sum(p.hi, p.lo);
}

// Exact for |i| < 2^48, which covers any index a kernel can be dispatched on.
inline df32 df_from_long(long i) {
  const float hi = static_cast<float>(i);
  return df32{hi, static_cast<float>(i - static_cast<long>(hi))};
}

} // namespace metal
} // namespace c10
