#pragma once

#include <c10/util/Half.h>


namespace std {

template <>
class numeric_limits<std::complex<float>> : public numeric_limits<float>  {};

template <>
class numeric_limits<std::complex<double>> : public numeric_limits<double>  {};

template <>
class numeric_limits<c10::ComplexHalf> : public numeric_limits<c10::Half>  {};

} // namespace std
