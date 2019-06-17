#pragma once

#include <ATen/core/Tensor.h>
#include <c10/macros/Macros.h>
#include <c10/util/Half.h>
#include <c10/util/Exception.h>
#include <ATen/core/DeprecatedTypeProperties.h>

#define AT_PRIVATE_CASE_TYPE(enum_type, type, ...) \
  case enum_type: {                                \
    using scalar_t = type;                         \
    return __VA_ARGS__();                          \
  }

#define AT_QINT_PRIVATE_CASE_TYPE(enum_type, type, underlying_enum, underlying_type, ...) \
  case enum_type: {                                                     \
    const auto& UNDERLYING_TYPE C10_UNUSED = underlying_enum;           \
    using scalar_t C10_UNUSED = type;                                   \
    using underlying_t C10_UNUSED = underlying_type;                    \
    return __VA_ARGS__();                                               \
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
struct ScalarTypeToCType<at::ScalarType::BFloat16> {
  using type = at::BFloat16;

  // This is a workaround for the CUDA bug which prevents ::detail::ScalarTypeToCType<T>::type being used directly
  // due to ambiguous reference which can't to be resolved. For some reason it cant pick between at::detail and at::cuda::detail.
  // For repro example, please see: https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
  // TODO: remove once the bug is fixed.
  static at::BFloat16 t;
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

// The AT_DISPATCH_* family of macros provides the ability to
// conveniently generate specializations of a kernel over all of the
// dtypes we care about in PyTorch.  We call it "dispatch" because
// we are "dispatching" to the correct, dtype-specific kernel.
//
// A standard usage looks like:
//
//      AT_DISPATCH_ALL_TYPES(self.scalar_type(), "op_name", [&] {
//          // Your code here, with 'scalar_t' now defined to
//          // be the dtype in question
//      })
//
// There are many variations of this macro, so it's important to
// understand exactly /which/ dtypes you want to get instantiated, as
// well as what the "default" set is.
//
// The default set of dtypes that are instantiated (e.g., by
// AT_DISPATCH_ALL_TYPES) are floating point types (float, double),
// and integral types (int32_t, int64_t, int16_t, int8_t, uint8_t),
// but NOT booleans (bool), half-precision floats (Half) or
// complex number (std::complex<float>, std::complex<double>).
// This "cut" is somewhat historical (the default types are the
// ones that TH historically supported), but it also reflects the
// fact that the non-default types are "poorly" behaved (booleans
// are NOT integers mod 2, half precision operations ~essentially
// don't exist on CPU, complex numbers are an experimental application).
//
// Here are the questions you should generally ask to decide which
// dispatch you want:
//
// 1. Is this an integral or floating point specific operation?
//    (If so, you'll want one of the FLOATING or INTEGRAL macros.)
//
// 2. Should half be supported?  (If you're on CPU, the answer is almost
//    definitely no.  If you do want support, use one of the AND_HALF
//    macros)
//
// Much rarer situations:
//
// 3. Should bool be supported?  (You often have to write your kernel
//    differently if arithmetic operations are involved.)  If so,
//    Use AT_DISPATCH_ALL_TYPES_AND along with ScalarType::Bool
//
// 4. Should complex be supported?  The answer is almost always no,
//    unless you are working on "generic" code that should work on
//    all dtypes.


// NB: the the_type variable is not used, but we have kept it for
// backwards compatibility.  It's probably not used by anyone though;
// but we're just being safe (and it doesn't hurt.)  Note we must
// use it to shut up warnings about unused store.

#define AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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

#define AT_DISPATCH_ALL_TYPES(TYPE, NAME, ...)                               \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op  */ \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)    \
      AT_PRIVATE_CASE_TYPE(                                                  \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)  \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

#define AT_DISPATCH_QINT_TYPES(TYPE, NAME, ...)                         \
  [&] {                                                                 \
    const auto& SCALAR_TYPE C10_UNUSED = TYPE;                          \
    switch (TYPE) {                                                     \
      AT_QINT_PRIVATE_CASE_TYPE(                                        \
          kQInt8, qint8, kChar, int8_t, __VA_ARGS__)                    \
      AT_QINT_PRIVATE_CASE_TYPE(                                        \
          kQUInt8, quint8, kByte, uint8_t, __VA_ARGS__)                 \
      AT_QINT_PRIVATE_CASE_TYPE(                                        \
          kQInt32, qint32, kInt, int, __VA_ARGS__)                      \
      default:                                                          \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'"); \
    }                                                                   \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX(TYPE, NAME, ...)                   \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op*/   \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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

#define AT_DISPATCH_ALL_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)                  \
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
      AT_PRIVATE_CASE_TYPE(SCALARTYPE3, decltype(::detail::ScalarTypeToCType<SCALARTYPE3>::t), __VA_ARGS__) \
      default:                                                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(TYPE), "'");                                     \
    }                                                                                                       \
  }()

#define AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, TYPE, NAME, ...)       \
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
      AT_PRIVATE_CASE_TYPE(SCALARTYPE3, decltype(::detail::ScalarTypeToCType<SCALARTYPE3>::t), __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(                                                                                 \
          at::ScalarType::ComplexFloat, std::complex<float>, __VA_ARGS__)                                   \
      AT_PRIVATE_CASE_TYPE(                                                                                 \
          at::ScalarType::ComplexDouble, std::complex<double>, __VA_ARGS__)                                 \
      default:                                                                                              \
        AT_ERROR(#NAME, " not implemented for '", TYPE, "'");                                               \
    }                                                                                                       \
  }()

// ----------------------------------------------------------------------------
// DEPRECATED MACROS, DON'T USE THESE
// ----------------------------------------------------------------------------

#define AT_DISPATCH_ALL_TYPES_AND_HALF(TYPE, NAME, ...)                      \
  [&] {                                                                      \
    detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF();                     \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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

#define AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX(TYPE, NAME, ...)          \
  [&] {                                                                      \
    detail::deprecated_AT_DISPATCH_ALL_TYPES_AND_HALF_AND_COMPLEX()          \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
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
