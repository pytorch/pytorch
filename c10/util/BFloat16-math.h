#pragma once

#include <c10/util/BFloat16-inl.h>
#include <c10/util/math_compat.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

namespace std {

/// Used by vec256<c10::BFloat16>::map
inline c10::BFloat16 acos(c10::BFloat16 a) {
  return std::acos(float(a));
}
inline c10::BFloat16 asin(c10::BFloat16 a) {
  return std::asin(float(a));
}
inline c10::BFloat16 atan(c10::BFloat16 a) {
  return std::atan(float(a));
}
inline c10::BFloat16 erf(c10::BFloat16 a) {
  return std::erf(float(a));
}
inline c10::BFloat16 erfc(c10::BFloat16 a) {
  return std::erfc(float(a));
}
inline c10::BFloat16 exp(c10::BFloat16 a) {
  return std::exp(float(a));
}
inline c10::BFloat16 expm1(c10::BFloat16 a) {
  return std::expm1(float(a));
}
inline c10::BFloat16 log(c10::BFloat16 a) {
  return std::log(float(a));
}
inline c10::BFloat16 log10(c10::BFloat16 a) {
  return std::log10(float(a));
}
inline c10::BFloat16 log1p(c10::BFloat16 a) {
  return std::log1p(float(a));
}
inline c10::BFloat16 log2(c10::BFloat16 a) {
  return std::log2(float(a));
}
inline c10::BFloat16 ceil(c10::BFloat16 a) {
  return std::ceil(float(a));
}
inline c10::BFloat16 cos(c10::BFloat16 a) {
  return std::cos(float(a));
}
inline c10::BFloat16 floor(c10::BFloat16 a) {
  return std::floor(float(a));
}
inline c10::BFloat16 nearbyint(c10::BFloat16 a) {
  return std::nearbyint(float(a));
}
inline c10::BFloat16 sin(c10::BFloat16 a) {
  return std::sin(float(a));
}
inline c10::BFloat16 tan(c10::BFloat16 a) {
  return std::tan(float(a));
}
inline c10::BFloat16 sinh(c10::BFloat16 a) {
  return std::sinh(float(a));
}
inline c10::BFloat16 cosh(c10::BFloat16 a) {
  return std::cosh(float(a));
}
inline c10::BFloat16 tanh(c10::BFloat16 a) {
  return std::tanh(float(a));
}
inline c10::BFloat16 trunc(c10::BFloat16 a) {
  return std::trunc(float(a));
}
inline c10::BFloat16 lgamma(c10::BFloat16 a) {
  return std::lgamma(float(a));
}
inline c10::BFloat16 sqrt(c10::BFloat16 a) {
  return std::sqrt(float(a));
}
inline c10::BFloat16 rsqrt(c10::BFloat16 a) {
  return 1.0 / std::sqrt(float(a));
}
inline c10::BFloat16 abs(c10::BFloat16 a) {
  return std::abs(float(a));
}
#if defined(_MSC_VER) && defined(__CUDACC__)
inline c10::BFloat16 pow(c10::BFloat16 a, double b) {
  return std::pow(float(a), float(b));
}
#else
inline c10::BFloat16 pow(c10::BFloat16 a, double b) {
  return std::pow(float(a), b);
}
#endif
inline c10::BFloat16 pow(c10::BFloat16 a, c10::BFloat16 b) {
  return std::pow(float(a), float(b));
}
inline c10::BFloat16 fmod(c10::BFloat16 a, c10::BFloat16 b) {
  return std::fmod(float(a), float(b));
}

/*
  The following function is inspired from the implementation in `musl`
  Link to License: https://git.musl-libc.org/cgit/musl/tree/COPYRIGHT
  ----------------------------------------------------------------------
  Copyright © 2005-2020 Rich Felker, et al.

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
  ----------------------------------------------------------------------
 */
C10_HOST_DEVICE inline c10::BFloat16 nextafter(
    c10::BFloat16 from,
    c10::BFloat16 to) {
  // Reference:
  // https://git.musl-libc.org/cgit/musl/tree/src/math/nextafter.c
  using int_repr_t = uint16_t;
  using float_t = c10::BFloat16;
  constexpr uint8_t bits = 16;
  union {
    float_t f;
    int_repr_t i;
  } ufrom = {from}, uto = {to};

  // get a mask to get the sign bit i.e. MSB
  int_repr_t sign_mask = int_repr_t{1} << (bits - 1);

  // short-circuit: if either is NaN, return NaN
  if (from != from || to != to) {
    return from + to;
  }

  // short-circuit: if they are exactly the same.
  if (ufrom.i == uto.i) {
    return from;
  }

  // mask the sign-bit to zero i.e. positive
  // equivalent to abs(x)
  int_repr_t abs_from = ufrom.i & ~sign_mask;
  int_repr_t abs_to = uto.i & ~sign_mask;
  if (abs_from == 0) {
    // if both are zero but with different sign,
    // preserve the sign of `to`.
    if (abs_to == 0) {
      return to;
    }
    // smallest subnormal with sign of `to`.
    ufrom.i = (uto.i & sign_mask) | int_repr_t{1};
    return ufrom.f;
  }

  // if abs(from) > abs(to) or sign(from) != sign(to)
  if (abs_from > abs_to || ((ufrom.i ^ uto.i) & sign_mask)) {
    ufrom.i--;
  } else {
    ufrom.i++;
  }

  return ufrom.f;
}

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
