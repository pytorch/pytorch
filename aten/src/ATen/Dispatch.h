#pragma once

#include <ATen/Type.h>
#include <c10/util/Half.h>
#include <c10/util/Exception.h>
#include <ATen/core/DeprecatedTypeProperties.h>

#define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                \
    using scalar_t = type;                         \
    return __VA_ARGS__();                          \
  }

namespace detail {

template <at::ScalarType N>
struct ScalarTypeToCType;

template<>
struct ScalarTypeToCType<at::ScalarType::Half> {
  using type = at::Half;

  // This is a workaround for the CUDA bug which prevents ::detail::ScalarTypeToCType<T>::type being used directly
  // due to ambiguous reference which can't to be resolved. For some reason it cant pick between at::detail and at::cuda::detail.
  // For repro example, please see: https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
  // TODO: remove once the bug is fixed.
  static at::Half t;
};

template<>
struct ScalarTypeToCType<at::ScalarType::Bool> {
  using type = bool;

  // This is a workaround for the CUDA bug which prevents ::detail::ScalarTypeToCType<T>::type being used directly
  // due to ambiguous reference which can't to be resolved. For some reason it cant pick between at::detail and at::cuda::detail.
  // For repro example, please see: https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
  // TODO: remove once the bug is fixed.
  static bool t;
};

inline at::ScalarType scalar_type(at::ScalarType s) {
  return s;
}

C10_DEPRECATED_MESSAGE("passing at::DeprecatedTypeProperties to an AT_DISPATCH macro is deprecated, " \
                       "pass an at::ScalarType instead")
inline at::ScalarType scalar_type(const at::DeprecatedTypeProperties &t) {
  return t.scalarType();
}

C10_DEPRECATED_MESSAGE("AT_DISPATCH_ALL_TYPES_AND_HALF is deprecated, " \
                       "use AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, ...) instead")
inline void deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF() {}

C10_DEPRECATED_MESSAGE("AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX is deprecated, "            \
                       "use AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Half, ...) " \
                       "instead")
inline void deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX() {}

}

// NB: the the_type variable is not used, but we have kept it for
// backwards compatibility.  It's probably not used by anyone though;
// but we're just being safe (and it doesn't hurt.)  Note we must
// use it to shut up warnings about unused store.

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...)                 \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(TYPE, NAME, ...)              \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexHalf, std::complex<at::Half>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, ...)                      \
  [&] {                                                                      \
    detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF();                     \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_COMPLEX_TYPES(TYPE, NAME, ...)                           \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)                   \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'"); \
    }                                                                        \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX(TYPE, NAME, ...)          \
  [&] {                                                                      \
    detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX()          \
    const auto& the_type = TYPE;                                             \
    (void)the_type;                                                          \
    at::ScalarType _st = ::detail::scalar_type(TYPE);                        \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, at::Half, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_ALL_TYPES_AND(SCALARTYPE, TYPE, NAME, ...)                                            \
  [&] {                                                                                                   \
    switch (TYPE) {                                                                                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                    \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                   \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                    \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                   \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE, decltype(::detail::ScalarTypeToCType<SCALARTYPE>::t), __VA_ARGS__) \
      default:                                                                                            \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                   \
    }                                                                                                     \
  }()

#define AT_DISPATCH_ALL_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                               \
  [&] {                                                                                                     \
    switch (TYPE) {                                                                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                     \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(::detail::ScalarTypeToCType<SCALARTYPE1>::t), __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(::detail::ScalarTypeToCType<SCALARTYPE2>::t), __VA_ARGS__) \
      default:                                                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                     \
    }                                                                                                       \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE1, SCALARTYPE2, TYPE, NAME, ...)                    \
  [&] {                                                                                                     \
    switch (TYPE) {                                                                                         \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Byte, uint8_t, __VA_ARGS__)                                      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, int8_t, __VA_ARGS__)                                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)                                     \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)                                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int32_t, __VA_ARGS__)                                       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)                                      \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Short, int16_t, __VA_ARGS__)                                     \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE1, decltype(::detail::ScalarTypeToCType<SCALARTYPE1>::t), __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(SCALARTYPE2, decltype(::detail::ScalarTypeToCType<SCALARTYPE2>::t), __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(                                                                                 \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                                   \
      AT_PRIVATE_CASE_TYPE(                                                                                 \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                                 \
      default:                                                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");                                               \
    }                                                                                                       \
  }()
