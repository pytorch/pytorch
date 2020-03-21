#pragma once

#include <c10/util/Half.h>


namespace std {

template <>
class numeric_limits<std::complex<float>> : public numeric_limits<float>  {};

template <>
class numeric_limits<std::complex<double>> : public numeric_limits<double>  {};

template <>
class numeric_limits<c10::ComplexHalf> : public numeric_limits<c10::Half>  {};

#define COMPLEX_INTEGER_OP_TEMPLATE_CONDITION \
  typename std::enable_if_t<std::is_floating_point<fT>::value && std::is_integral<iT>::value, int> = 0

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator+(const std::complex<fT>& a, const iT& b) { return a + static_cast<fT>(b); }

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator+(const iT& a, const std::complex<fT>& b) { return static_cast<fT>(a) + b; }

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator-(const std::complex<fT>& a, const iT& b) { return a - static_cast<fT>(b); }

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator-(const iT& a, const std::complex<fT>& b) { return static_cast<fT>(a) - b; }

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator*(const std::complex<fT>& a, const iT& b) { return a * static_cast<fT>(b); }

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator*(const iT& a, const std::complex<fT>& b) { return static_cast<fT>(a) * b; }

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator/(const std::complex<fT>& a, const iT& b) { return a / static_cast<fT>(b); }

template<typename fT, typename iT, COMPLEX_INTEGER_OP_TEMPLATE_CONDITION>
std::complex<fT> operator/(const iT& a, const std::complex<fT>& b) { return static_cast<fT>(a) / b; }

#undef COMPLEX_INTEGER_OP_TEMPLATE_CONDITION
} // namespace std
