#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>

namespace at {

// For FP16 or BFloat16 inputs, ops should perform internal math in FP32.
template <typename scalar_t, bool disable_opmath_type = false>
struct OpMathType {
  using type = scalar_t;
};
template <>
struct OpMathType<at::Half, false> {
  using type = float;
};
template <>
struct OpMathType<at::BFloat16, false> {
  using type = float;
};
template <>
struct OpMathType<c10::complex<Half>, false> {
  using type = c10::complex<float>;
};

template <typename T, bool disable_opmath_type = false>
using opmath_type = typename OpMathType<T, disable_opmath_type>::type;

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
