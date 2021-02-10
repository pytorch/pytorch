#pragma once

#include <c10/util/BFloat16-inl.h>
#include <c10/util/math_compat.h>

namespace std {

/// Used by vec256<c10::BFloat16>::map
inline c10::BFloat16 acos(c10::BFloat16 a) { return std::acos(float(a));}
inline c10::BFloat16 asin(c10::BFloat16 a) { return std::asin(float(a));}
inline c10::BFloat16 atan(c10::BFloat16 a) { return std::atan(float(a));}
inline c10::BFloat16 erf(c10::BFloat16 a) { return std::erf(float(a));}
inline c10::BFloat16 erfc(c10::BFloat16 a) { return std::erfc(float(a));}
inline c10::BFloat16 exp(c10::BFloat16 a) { return std::exp(float(a));}
inline c10::BFloat16 expm1(c10::BFloat16 a) { return std::expm1(float(a));}
inline c10::BFloat16 log(c10::BFloat16 a) { return std::log(float(a));}
inline c10::BFloat16 log10(c10::BFloat16 a) { return std::log10(float(a));}
inline c10::BFloat16 log1p(c10::BFloat16 a) { return std::log1p(float(a));}
inline c10::BFloat16 log2(c10::BFloat16 a) { return std::log2(float(a));}
inline c10::BFloat16 ceil(c10::BFloat16 a) { return std::ceil(float(a));}
inline c10::BFloat16 cos(c10::BFloat16 a) { return std::cos(float(a));}
inline c10::BFloat16 floor(c10::BFloat16 a) { return std::floor(float(a));}
inline c10::BFloat16 nearbyint(c10::BFloat16 a) { return std::nearbyint(float(a));}
inline c10::BFloat16 sin(c10::BFloat16 a) { return std::sin(float(a));}
inline c10::BFloat16 tan(c10::BFloat16 a) { return std::tan(float(a));}
inline c10::BFloat16 tanh(c10::BFloat16 a) { return std::tanh(float(a));}
inline c10::BFloat16 trunc(c10::BFloat16 a) { return std::trunc(float(a));}
inline c10::BFloat16 lgamma(c10::BFloat16 a) { return std::lgamma(float(a));}
inline c10::BFloat16 sqrt(c10::BFloat16 a) { return std::sqrt(float(a));}
inline c10::BFloat16 rsqrt(c10::BFloat16 a) { return 1.0 / std::sqrt(float(a));}
inline c10::BFloat16 abs(c10::BFloat16 a) { return std::abs(float(a));}
inline c10::BFloat16 min(c10::BFloat16 a, c10::BFloat16 b) { return std::min(float(a), float(b));}
inline c10::BFloat16 max(c10::BFloat16 a, c10::BFloat16 b) { return std::max(float(a), float(b));}
#if defined(_MSC_VER) && defined(__CUDACC__)
inline c10::BFloat16 pow(c10::BFloat16 a, double b) { return std::pow(float(a), float(b));}
#else
inline c10::BFloat16 pow(c10::BFloat16 a, double b) { return std::pow(float(a), b);}
#endif
inline c10::BFloat16 pow(c10::BFloat16 a, c10::BFloat16 b) { return std::pow(float(a), float(b));}
inline c10::BFloat16 fmod(c10::BFloat16 a, c10::BFloat16 b) { return std::fmod(float(a), float(b));}

} // namespace std
