#pragma once

#include <complex>


namespace std {

template <typename T> struct is_complex_t                  : public std::false_type {};
template <typename T> struct is_complex_t<std::complex<T>> : public std::true_type {};

template <>
class numeric_limits<std::complex<float>> : public numeric_limits<float>  {};

template <>
class numeric_limits<std::complex<double>> : public numeric_limits<double>  {};

} // namespace std
