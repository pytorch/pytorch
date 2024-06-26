#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>

namespace at {

// For FP16 or BFloat16 inputs, ops should perform internal math in FP32.
template <typename scalar_t>
struct OpMathType {
  using type = scalar_t;
};
template <>
struct OpMathType<at::Half> {
  using type = float;
};
template <>
struct OpMathType<at::BFloat16> {
  using type = float;
};
template <>
struct OpMathType<at::Float8_e5m2> {
  using type = float;
};
template <>
struct OpMathType<at::Float8_e4m3fn> {
  using type = float;
};
template <>
struct OpMathType<at::Float8_e5m2fnuz> {
  using type = float;
};
template <>
struct OpMathType<at::Float8_e4m3fnuz> {
  using type = float;
};
template <>
struct OpMathType<c10::complex<Half>> {
  using type = c10::complex<float>;
};

template <typename T>
using opmath_type = typename OpMathType<T>::type;

namespace {

inline c10::ScalarType toOpMathType(const c10::ScalarType type) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum) \
  case ScalarType::TypeNum:            \
    return CppTypeToScalarType<at::opmath_type<scalar_t>>::value;

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
#undef DEFINE_CASE

    default:
      TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

} // namespace

} // namespace at
