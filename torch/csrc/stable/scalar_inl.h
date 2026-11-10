#pragma once

#include <torch/csrc/stable/scalar_struct.h>
#include <torch/csrc/stable/stableivalue_conversions.h>

HIDDEN_NAMESPACE_BEGIN(torch, stable)

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

inline Scalar::Scalar(double value) {
  AtenScalarHandle scalar = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_new_scalar_double(value, &scalar));
  *this = Scalar(scalar);
}

inline Scalar::Scalar(bool value) {
  AtenScalarHandle scalar = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_new_scalar_bool(value, &scalar));
  *this = Scalar(scalar);
}

inline Scalar::Scalar(c10::complex<double> value) {
  AtenScalarHandle scalar = nullptr;
  STABLE_TORCH_ERROR_CODE_CHECK(
      torch_new_scalar_complex(value.real(), value.imag(), &scalar));
  *this = Scalar(scalar);
}

inline torch::headeronly::ScalarType Scalar::type() const {
  int32_t dtype = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_type(get(), &dtype));
  return detail::to<torch::headeronly::ScalarType>(detail::from(dtype));
}

inline bool Scalar::isBoolean() const {
  return type() == torch::headeronly::ScalarType::Bool;
}

inline bool Scalar::isIntegral(bool includeBool) const {
  const auto dtype = type();
  return dtype == torch::headeronly::ScalarType::Long ||
      (includeBool && dtype == torch::headeronly::ScalarType::Bool);
}

inline bool Scalar::isFloatingPoint() const {
  return type() == torch::headeronly::ScalarType::Double;
}

inline bool Scalar::isComplex() const {
  return type() == torch::headeronly::ScalarType::ComplexDouble;
}

inline int64_t Scalar::toLong() const {
  int64_t value = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_to_int(get(), &value));
  return value;
}

inline double Scalar::toDouble() const {
  double value = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_to_double(get(), &value));
  return value;
}

inline bool Scalar::toBool() const {
  bool value = false;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_to_bool(get(), &value));
  return value;
}

inline c10::complex<double> Scalar::toComplexDouble() const {
  double real = 0;
  double imag = 0;
  STABLE_TORCH_ERROR_CODE_CHECK(torch_scalar_to_complex(get(), &real, &imag));
  return {real, imag};
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

HIDDEN_NAMESPACE_END(torch, stable)
