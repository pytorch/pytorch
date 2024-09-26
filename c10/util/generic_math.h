#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/MathConstants.h>
#include <c10/util/TypeSafeSignMath.h>
#include <array>
#include <cmath>

#if defined(__CUDA_ARCH__)
#include <c10/cuda/CUDAMathCompat.h>
#define C10_COMPAT_COPYSIGN c10::cuda::compat::copysign
#elif defined(__HIPCC__)
#include <c10/hip/HIPMathCompat.h>
#define C10_COMPAT_COPYSIGN c10::hip::compat::copysign
#else
#include <c10/util/copysign.h>
#define C10_COMPAT_COPYSIGN c10::copysign
#endif

// The functions in this file should be header-only as it is used under
// ABI-compatibility mode.

namespace c10 {

// NOTE: [Floor Division in Python]
// Python's __floordiv__ operator is more complicated than just floor(a / b).
// It aims to maintain the property: a == (a // b) * b + remainder(a, b)
// which can otherwise fail due to rounding errors in the remainder.
// So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
// With some additional fix-ups added to the result.
//
// For reference, see CPython's implementation:
// https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_floating(scalar_t a, scalar_t b)
    __ubsan_ignore_float_divide_by_zero__ {
  if (C10_UNLIKELY(b == 0)) {
    // Divide by zero: return standard IEEE result
    return a / b;
  }

  auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && (b < 0) != (mod < 0)) {
    div -= scalar_t(1);
  }

  scalar_t floordiv;
  if (div != 0) {
    floordiv = std::floor(div);
    if (div - floordiv > scalar_t(0.5)) {
      floordiv += scalar_t(1.0);
    }
  } else {
    floordiv = C10_COMPAT_COPYSIGN(scalar_t(0), a / b);
  }
  return floordiv;
}

template <typename scalar_t>
inline C10_HOST_DEVICE scalar_t div_floor_integer(scalar_t a, scalar_t b) {
  if (c10::signs_differ(a, b)) {
    // Subtracts one from the results of truncation division if the
    // divisor and dividend have different sign(bit)s and the remainder of
    // the division is nonzero
    const auto quot = a / b;
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  }
  return a / b;
}

#define CENTRAL_RANGE 0.7

template <typename T>
inline typename std::enable_if<std::is_floating_point<T>::value, T>::type
calc_erfinv(T y) {
  /* Function to calculate inverse error function.  Rational approximation
  is used to generate an initial approximation, which is then improved to
  full accuracy by two steps of Newton's method.  Code is a direct
  translation of the erfinv m file in matlab version 2.0.
  Author:  Gary L. Pavlis, Indiana University
  Date:  February 1996
  */
  T x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */
  std::array<T, 4> a = {
      T(0.886226899), T(-1.645349621), T(0.914624893), T(-0.140543331)};
  std::array<T, 4> b = {
      T(-2.118377725), T(1.442710462), T(-0.329097515), T(0.012229801)};
  std::array<T, 4> c = {
      T(-1.970840454), T(-1.624906493), T(3.429567803), T(1.641345311)};
  std::array<T, 2> d = {T(3.543889200), T(1.637067800)};
  T y_abs = std::abs(y);
  if (y_abs > 1.0) {
    return std::numeric_limits<T>::quiet_NaN();
  }
#ifdef _WIN32
  // error C2039: '_copysign': is not a member of 'std'
  if (y_abs == 1.0) {
    return copysign(std::numeric_limits<T>::infinity(), y);
  }
#else
  if (y_abs == 1.0) {
    return std::copysign(std::numeric_limits<T>::infinity(), y);
  }
#endif
  if (y_abs <= static_cast<T>(CENTRAL_RANGE)) {
    z = y * y;
    num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
    dem =
        ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + static_cast<T>(1.0));
    x = y * num / dem;
  } else {
    z = std::sqrt(
        -std::log((static_cast<T>(1.0) - y_abs) / static_cast<T>(2.0)));
    num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
    dem = (d[1] * z + d[0]) * z + static_cast<T>(1.0);
#ifdef _WIN32
    // error C2039: '_copysign': is not a member of 'std'
    x = copysign(num, y) / dem;
#else
    x = std::copysign(num, y) / dem;
#endif
  }
  /* Two steps of Newton-Raphson correction */
  x = x -
      (std::erf(x) - y) /
          ((static_cast<T>(2.0) / static_cast<T>(std::sqrt(c10::pi<double>))) *
           std::exp(-x * x));
  x = x -
      (std::erf(x) - y) /
          ((static_cast<T>(2.0) / static_cast<T>(std::sqrt(c10::pi<double>))) *
           std::exp(-x * x));

  return (x);
}

#undef CENTRAL_RANGE

inline c10::BFloat16 calc_erfinv(c10::BFloat16 a) {
  return calc_erfinv(float(a));
}

} // namespace c10
