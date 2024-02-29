#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-float-conversion")
#endif

namespace std {

template <typename T>
struct is_reduced_floating_point
    : std::integral_constant<
          bool,
          std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16>> {};

template <typename T>
constexpr bool is_reduced_floating_point_v =
    is_reduced_floating_point<T>::value;

template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T acos(T a) {
  return std::acos(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T asin(T a) {
  return std::asin(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T atan(T a) {
  return std::atan(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T atanh(T a) {
  return std::atanh(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T erf(T a) {
  return std::erf(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T erfc(T a) {
  return std::erfc(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T exp(T a) {
  return std::exp(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T expm1(T a) {
  return std::expm1(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log(T a) {
  return std::log(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log10(T a) {
  return std::log10(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log1p(T a) {
  return std::log1p(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T log2(T a) {
  return std::log2(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T ceil(T a) {
  return std::ceil(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T cos(T a) {
  return std::cos(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T floor(T a) {
  return std::floor(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T nearbyint(T a) {
  return std::nearbyint(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T sin(T a) {
  return std::sin(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T tan(T a) {
  return std::tan(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T sinh(T a) {
  return std::sinh(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T cosh(T a) {
  return std::cosh(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T tanh(T a) {
  return std::tanh(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T trunc(T a) {
  return std::trunc(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T lgamma(T a) {
  return std::lgamma(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T sqrt(T a) {
  return std::sqrt(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T rsqrt(T a) {
  return 1.0 / std::sqrt(float(a));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T abs(T a) {
  return std::abs(float(a));
}
#if defined(_MSC_VER) && defined(__CUDACC__)
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T pow(T a, double b) {
  return std::pow(float(a), float(b));
}
#else
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T pow(T a, double b) {
  return std::pow(float(a), b);
}
#endif
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T pow(T a, T b) {
  return std::pow(float(a), float(b));
}
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
inline T fmod(T a, T b) {
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
template <
    typename T,
    typename std::enable_if_t<is_reduced_floating_point_v<T>, int> = 0>
C10_HOST_DEVICE inline T nextafter(T from, T to) {
  // Reference:
  // https://git.musl-libc.org/cgit/musl/tree/src/math/nextafter.c
  using int_repr_t = uint16_t;
  using float_t = T;
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
