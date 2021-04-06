#pragma once

#include <c10/util/Half.h>
#include <c10/util/math_compat.h>

namespace std {

/// Used by vec256<c10::Half>::map
inline c10::Half acos(c10::Half a) { return std::acos(float(a));}
inline c10::Half asin(c10::Half a) { return std::asin(float(a));}
inline c10::Half atan(c10::Half a) { return std::atan(float(a));}
inline c10::Half erf(c10::Half a) { return std::erf(float(a));}
inline c10::Half erfc(c10::Half a) { return std::erfc(float(a));}
inline c10::Half exp(c10::Half a) { return std::exp(float(a));}
inline c10::Half expm1(c10::Half a) { return std::expm1(float(a));}
inline c10::Half log(c10::Half a) { return std::log(float(a));}
inline c10::Half log10(c10::Half a) { return std::log10(float(a));}
inline c10::Half log1p(c10::Half a) { return std::log1p(float(a));}
inline c10::Half log2(c10::Half a) { return std::log2(float(a));}
inline c10::Half ceil(c10::Half a) { return std::ceil(float(a));}
inline c10::Half cos(c10::Half a) { return std::cos(float(a));}
inline c10::Half floor(c10::Half a) { return std::floor(float(a));}
inline c10::Half nearbyint(c10::Half a) { return std::nearbyint(float(a));}
inline c10::Half sin(c10::Half a) { return std::sin(float(a));}
inline c10::Half tan(c10::Half a) { return std::tan(float(a));}
inline c10::Half tanh(c10::Half a) { return std::tanh(float(a));}
inline c10::Half trunc(c10::Half a) { return std::trunc(float(a));}
inline c10::Half lgamma(c10::Half a) { return std::lgamma(float(a));}
inline c10::Half sqrt(c10::Half a) { return std::sqrt(float(a));}
inline c10::Half rsqrt(c10::Half a) { return 1.0 / std::sqrt(float(a));}
inline c10::Half abs(c10::Half a) { return std::abs(float(a));}
inline c10::Half min(c10::Half a, c10::Half b) { return std::min(float(a), float(b));}
inline c10::Half max(c10::Half a, c10::Half b) { return std::max(float(a), float(b));}
inline c10::Half pow(c10::Half a, c10::Half b) { return std::pow(float(a), float(b));}
inline c10::Half fmod(c10::Half a, c10::Half b) { return std::fmod(float(a), float(b));}
inline c10::Half copysign(c10::Half a, c10::Half b) {
  return c10::Half((a.x&0x7fff) | (b.x&0x8000), c10::Half::from_bits());
}

}
