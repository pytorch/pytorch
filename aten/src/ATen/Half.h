#pragma once

#include "ATen/ATenGeneral.h"

#include <limits>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <cmath>

namespace at {

template<typename To, typename From> To convert(From f) {
  return static_cast<To>(f);
}

// skip isnan and isinf check for integral types
template<typename To, typename From>
typename std::enable_if<std::is_integral<From>::value, bool>::type overflows(From f) {
  using limit = std::numeric_limits<To>;
  return f < limit::lowest() || f > limit::max();
}

template<typename To, typename From>
typename std::enable_if<!std::is_integral<From>::value, bool>::type overflows(From f) {
  using limit = std::numeric_limits<To>;
  if (limit::has_infinity && std::isinf(f)) {
    return false;
  }
  if (!limit::has_quiet_NaN && std::isnan(f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

template<typename To, typename From> To checked_convert(From f, const char* name) {
  if (overflows<To, From>(f)) {
    std::string msg = "value cannot be converted to type ";
    msg += name;
    msg += " without overflow: ";
    msg += std::to_string(f);
    throw std::domain_error(std::move(msg));
  }
  return convert<To, From>(f);
}

struct alignas(2) Half {
  unsigned short x;
  operator double();
};

template<> AT_API Half convert(float f);
template<> AT_API float convert(Half f);
template<> AT_API Half convert(double f);
template<> AT_API double convert(Half f);
template<> AT_API Half convert(int64_t f);
template<> AT_API int64_t convert(Half f);

inline Half::operator double() {
  return convert<double, Half>(*this);
}

template<> bool overflows<Half, double>(double f);
template<> bool overflows<Half, int64_t>(int64_t f);

template<typename To, typename From>
To HalfFix(From h) {
  return To { h.x };
}
} // namespace at
