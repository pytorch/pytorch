#pragma once

#if !defined(C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H)
#error \
    "torch/headeronly/util/complex_utils.h is not meant to be individually included. Include torch/headeronly/util/complex.h instead."
#endif

#include <limits>

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

template <typename T>
struct is_complex : public std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : public std::true_type {};

template <typename T>
struct is_complex<c10::complex<T>> : public std::true_type {};

// Extract double from std::complex<double>; is identity otherwise
// TODO: Write in more idiomatic C++17
template <typename T>
struct scalar_value_type {
  using type = T;
};
template <typename T>
struct scalar_value_type<std::complex<T>> {
  using type = T;
};
template <typename T>
struct scalar_value_type<c10::complex<T>> {
  using type = T;
};

HIDDEN_NAMESPACE_END(torch, headeronly)

namespace std {

template <typename T>
class numeric_limits<c10::complex<T>> : public numeric_limits<T> {};

template <typename T>
bool isnan(const c10::complex<T>& v) {
  return std::isnan(v.real()) || std::isnan(v.imag());
}

} // namespace std
