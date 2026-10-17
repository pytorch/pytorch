// Double-float arithmetic library
#pragma once
#ifdef __METAL__
#include <metal_stdlib>
#else
#include <cmath>
#endif

namespace c10 {
namespace metal {

namespace detail {
inline float df_fma(float a, float b, float c) {
#ifdef __METAL__
  return ::metal::fma(a, b, c);
#else
  return std::fma(a, b, c);
#endif
}
} // namespace detail

// A value held as the unevaluated sum hi + lo of two float32s, with
// |lo| <= ulp(hi) / 2. It lets a kernel carry an intermediate at better than
// float32 precision where Metal Shading Language offers no `double`.
//
// How it compares with a real float64:
//   - The significand is about 48 bits, the two non-overlapping float32
//     significands, against 53 for float64. That is enough for rounding the
//     result down to float32 to land where a float64 computation would, but
//     it is not a float64 substitute.
//   - The exponent range is float32's, so the largest finite value is ~3.4e38
//     rather than ~1.8e308. This buys precision, never dynamic range.
//   - The extra precision disappears at the ends of that range: once |hi| is
//     subnormal the low word underflows to zero, and near the top of the
//     range hi + lo is no longer representable either.
//   - Results are not correctly rounded. add and mul are good to a few ulps
//     of the 48-bit format; div is looser still, since it refines a float32
//     quotient rather than computing an exact one.
//   - Infinities and NaNs are handled only as far as the underlying float32
//     operations handle them: hi carries what IEEE gives, while lo can come
//     out NaN.
//
// The error-free transformations below are exact under IEEE semantics and
// only under those: they depend on the compiler not reassociating them, which
// holds because the metallib is built with -fno-fast-math (cmake/Metal.cmake)
// and runtime compilation uses MTLMathModeSafe unless PYTORCH_MPS_FAST_MATH
// asks otherwise.
struct df32 {
  float hi;
  float lo;

  df32(float hi_, float lo_) : hi(hi_), lo(lo_) {}

  // Exact for |i| < 2^48, which covers any index a kernel is dispatched on.
  explicit df32(long i)
      : hi(static_cast<float>(i)),
        lo(static_cast<float>(i - static_cast<long>(static_cast<float>(i)))) {}

#ifndef __METAL__
  // Host-side split, for uploading a double as the pair a shader consumes.
  explicit df32(double v)
      : hi(static_cast<float>(v)),
        lo(std::isfinite(static_cast<float>(v))
               ? static_cast<float>(
                     v - static_cast<double>(static_cast<float>(v)))
               : 0.0f) {}
#endif
};

// Knuth's two-sum: the result is a + b exactly, for any a and b.
inline df32 two_sum(float a, float b) {
  const float s = a + b;
  const float b_virtual = s - a;
  return df32(s, (a - (s - b_virtual)) + (b - b_virtual));
}

// Dekker's fast two-sum. Same result as two_sum, but only when |a| >= |b|.
inline df32 quick_two_sum(float a, float b) {
  const float s = a + b;
  return df32(s, b - (s - a));
}

// Exact product: the residual is whatever the multiply rounded away, which
// fma() recovers.
inline df32 two_prod(float a, float b) {
  const float p = a * b;
  return df32(p, detail::df_fma(a, b, -p));
}

inline df32 neg(df32 a) {
  return df32(-a.hi, -a.lo);
}

inline df32 add(df32 a, df32 b) {
  df32 s = two_sum(a.hi, b.hi);
  s.lo += a.lo + b.lo;
  return quick_two_sum(s.hi, s.lo);
}

inline df32 sub(df32 a, df32 b) {
  return add(a, neg(b));
}

inline df32 mul(df32 a, df32 b) {
  df32 p = two_prod(a.hi, b.hi);
  p.lo += a.hi * b.lo + a.lo * b.hi;
  return quick_two_sum(p.hi, p.lo);
}

// Long division on the leading words, correcting the remainder twice.
inline df32 div(df32 a, df32 b) {
  const float q1 = a.hi / b.hi;
  df32 r = sub(a, mul(b, df32(q1, 0.0f)));
  const float q2 = r.hi / b.hi;
  r = sub(r, mul(b, df32(q2, 0.0f)));
  const float q3 = r.hi / b.hi;
  return add(quick_two_sum(q1, q2), df32(q3, 0.0f));
}

} // namespace metal
} // namespace c10
