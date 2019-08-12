#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>
#include <c10/util/typeid.h>

#include <complex>
#include <cstdint>
#include <iostream>

namespace c10 {

// For the macros below:
// NB: QInt ScalarTypes are referred to as "STUBS" here since they do not
// contain complete information to determine the tensor value of the data,
// they are just stubs for dispatch / quantization.
// NB: If you want to macro some code for all non-stub scalar types, you
// probably want one of the AT_FORALL_SCALAR_TYPES / AT_FORALL_SCALAR_TYPES_AND
// macros below, which are designed to behave similarly to the Dispatch macros
// with the same name.

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_STUBS(_)        \
  _(uint8_t, Byte, __) /* 0 */                                  \
  _(int8_t, Char, __) /* 1 */                                   \
  _(int16_t, Short, __) /* 2 */                                 \
  _(int, Int, __) /* 3 */                                       \
  _(int64_t, Long, __) /* 4 */                                  \
  _(at::Half, Half, __) /* 5 */                                 \
  _(float, Float, __) /* 6 */                                   \
  _(double, Double, __) /* 7 */                                 \
  _(at::ComplexHalf, ComplexHalf, __) /* 8 */                   \
  _(std::complex<float>, ComplexFloat, __) /* 9 */              \
  _(std::complex<double>, ComplexDouble, __) /* 10 */           \
  _(bool, Bool, __) /* 11 */                                    \
  _(c10::qint8, QInt8, __) /* 12 */                             \
  _(c10::quint8, QUInt8, __) /* 13 */                           \
  _(c10::qint32, QInt32, __) /* 14 */                           \
  _(at::BFloat16, BFloat16, __) /* 15 */


// If you want to support ComplexHalf for real, add ComplexHalf
// into this macro (and change the name).  But beware: convert()
// doesn't work for all the conversions you need...
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(_) \
  _(uint8_t, Byte, __)                                             \
  _(int8_t, Char, __)                                              \
  _(int16_t, Short, __)                                            \
  _(int, Int, __)                                                  \
  _(int64_t, Long, __)                                             \
  _(at::Half, Half, __)                                            \
  _(float, Float, __)                                              \
  _(double, Double, __)                                            \
  _(std::complex<float>, ComplexFloat, __)                         \
  _(std::complex<double>, ComplexDouble, __)                       \
  _(bool, Bool, __)                                                \
  _(at::BFloat16, BFloat16, __)


#define AT_FORALL_QINT_TYPES(_)  \
  _(c10::qint8, QInt8, i)        \
  _(c10::quint8, QUInt8, i)      \
  _(c10::qint32, QInt32, i)

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n, _2) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_STUBS(DEFINE_ENUM)
#undef DEFINE_ENUM
      Undefined,
  NumOptions
};

namespace impl {

// These are used to map ScalarTypes to C++ types.  Feel free to add more or even
// macro generate this; the examples here are just those we have found to be
// necessary.

template <c10::ScalarType N>
struct ScalarTypeToCPPType;

template<>
struct ScalarTypeToCPPType<c10::ScalarType::Half> {
  using type = c10::Half;

  // This is a workaround for the CUDA bug which prevents ::detail::ScalarTypeToCType<T>::type being used directly
  // due to ambiguous reference which can't to be resolved. For some reason it cant pick between at::detail and at::cuda::detail.
  // For repro example, please see: https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
  // TODO: remove once the bug is fixed.
  static type t;
};

template<>
struct ScalarTypeToCPPType<c10::ScalarType::BFloat16> {
  using type = c10::BFloat16;

  // This is a workaround for the CUDA bug which prevents ::detail::ScalarTypeToCType<T>::type being used directly
  // due to ambiguous reference which can't to be resolved. For some reason it cant pick between at::detail and at::cuda::detail.
  // For repro example, please see: https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
  // TODO: remove once the bug is fixed.
  static type t;
};

template<>
struct ScalarTypeToCPPType<c10::ScalarType::Bool> {
  using type = bool;

  // This is a workaround for the CUDA bug which prevents ::detail::ScalarTypeToCType<T>::type being used directly
  // due to ambiguous reference which can't to be resolved. For some reason it cant pick between at::detail and at::cuda::detail.
  // For repro example, please see: https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
  // TODO: remove once the bug is fixed.
  static type t;
};

template<>
struct ScalarTypeToCPPType<c10::ScalarType::Long> {
  using type = int64_t;

  // This is a workaround for the CUDA bug which prevents ::detail::ScalarTypeToCType<T>::type being used directly
  // due to ambiguous reference which can't to be resolved. For some reason it cant pick between at::detail and at::cuda::detail.
  // For repro example, please see: https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba
  // TODO: remove once the bug is fixed.
  static type t;
};
}

#define AT_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte, __)            \
  _(int8_t, Char, __)             \
  _(int16_t, Short, __)           \
  _(int, Int, __)                 \
  _(int64_t, Long, __)            \
  _(float, Float, __)             \
  _(double, Double, __)

#define AT_FORALL_SCALAR_TYPES_AND(SCALARTYPE, _)                              \
  _(uint8_t, Byte, __)                                                         \
  _(int8_t, Char, __)                                                          \
  _(int16_t, Short, __)                                                        \
  _(int, Int, __)                                                              \
  _(int64_t, Long, __)                                                         \
  _(at::Half, Half, __)                                                        \
  _(float, Float, __)                                                          \
  _(double, Double, __)                                                        \
  _(decltype(::c10::impl::ScalarTypeToCPPType<SCALARTYPE>::t), SCALARTYPE, __)

#define AT_FORALL_SCALAR_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _)                                  \
  _(uint8_t, Byte, __)                                                                            \
  _(int8_t, Char, __)                                                                             \
  _(int16_t, Short, __)                                                                           \
  _(int, Int, __)                                                                                 \
  _(int64_t, Long, __)                                                                            \
  _(at::Half, Half, __)                                                                           \
  _(float, Float, __)                                                                             \
  _(double, Double, __)                                                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<c10::ScalarType::SCALARTYPE1>::t), SCALARTYPE1, __) \
  _(decltype(::c10::impl::ScalarTypeToCPPType<c10::ScalarType::SCALARTYPE2>::t), SCALARTYPE2, __)

static inline caffe2::TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
#define DEFINE_CASE(ctype, name, _) \
  case ScalarType::name:            \
    return caffe2::TypeMeta::Make<ctype>();

  switch (scalar_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_STUBS(DEFINE_CASE)
    case ScalarType::Undefined:
      return caffe2::TypeMeta();
    default:
      AT_ERROR(
          "Unrecognized Scalartype ",
          scalar_type,
          " (please report this error)");
  }
#undef DEFINE_CASE
}

static inline c10::optional<ScalarType> tryTypeMetaToScalarType(
    caffe2::TypeMeta dtype) {
#define DEFINE_IF(ctype, name, _)                 \
  if (dtype == caffe2::TypeMeta::Make<ctype>()) { \
    return {ScalarType::name};                    \
  }
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_STUBS(DEFINE_IF)
#undef DEFINE_IF
  if (dtype == caffe2::TypeMeta()) {
    return {ScalarType::Undefined};
  }
  return c10::nullopt;
}

static inline ScalarType typeMetaToScalarType(caffe2::TypeMeta dtype) {
  if (auto scalar_type = tryTypeMetaToScalarType(dtype)) {
    return *scalar_type;
  }
  AT_ERROR(
      "Unsupported TypeMeta in ATen: ", dtype, " (please report this error)");
}

static inline bool operator==(ScalarType t, caffe2::TypeMeta m) {
  if (auto mt = tryTypeMetaToScalarType(m)) {
    return (*mt) == t;
  }
  return false;
}

static inline bool operator==(caffe2::TypeMeta m, ScalarType t) {
  return t == m;
}

#define DEFINE_CONSTANT(_, name, _2) \
  constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_STUBS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name, _2) \
  case ScalarType::name:         \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_STUBS(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

static inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name, _2) \
  case ScalarType::name:                       \
    return sizeof(ctype);

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_STUBS(CASE_ELEMENTSIZE_CASE)
    default:
      AT_ERROR("Unknown ScalarType");
  }
#undef CASE_ELEMENTSIZE_CASE
}

C10_DEPRECATED_MESSAGE("isIntegralType is deprecated. Please use the overload with 'includeBool' parameter instead.")
static inline bool isIntegralType(ScalarType t) {
  return (
      t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
      t == ScalarType::Long || t == ScalarType::Short);
}

static inline bool isIntegralType(ScalarType t, bool includeBool) {
  bool isIntegral = (
      t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
      t == ScalarType::Long || t == ScalarType::Short);

  return includeBool ? isIntegral || (t == ScalarType::Bool) : isIntegral;
}

static inline bool isFloatingType(ScalarType t) {
  return (
      t == ScalarType::Double || t == ScalarType::Float ||
      t == ScalarType::Half || t == ScalarType::BFloat16);
}

static inline bool isComplexType(ScalarType t) {
  return (
      t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat ||
      t == ScalarType::ComplexDouble);
}

static inline bool isQIntType(ScalarType t) {
  // Don't forget to extend this when adding new QInt types
  return t == ScalarType:: QInt8 || t == ScalarType::QUInt8 || t == ScalarType::QInt32;
}

static inline ScalarType toQIntType(ScalarType t) {
  switch (t) {
    case ScalarType::Byte:
      return ScalarType::QUInt8;
    case ScalarType::Char:
      return ScalarType::QInt8;
    case ScalarType::Int:
      return ScalarType::QInt32;
    default:
      return t;
  }
}

static inline ScalarType toUnderlying(ScalarType t) {
  switch (t) {
    case ScalarType::QUInt8:
      return ScalarType::Byte;
    case ScalarType::QInt8:
      return ScalarType::Char;
    case ScalarType::QInt32:
      return ScalarType::Int;
    default:
      return t;
  }
}

static inline bool isUnderlying(ScalarType type, ScalarType qtype) {
  return type == toUnderlying(qtype);
}

static inline ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // This is generated according to NumPy's promote_types
  constexpr auto u1 = ScalarType::Byte;
  constexpr auto i1 = ScalarType::Char;
  constexpr auto i2 = ScalarType::Short;
  constexpr auto i4 = ScalarType::Int;
  constexpr auto i8 = ScalarType::Long;
  constexpr auto f2 = ScalarType::Half;
  constexpr auto f4 = ScalarType::Float;
  constexpr auto f8 = ScalarType::Double;
  constexpr auto b1 = ScalarType::Bool;
  constexpr auto bf = ScalarType::BFloat16;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }
  if (isComplexType(a) || isComplexType(b)) {
    AT_ERROR(
        "promoteTypes with complex numbers is not handled yet; figure out what the correct rules should be");
  }

  // For QInt types, we only allow exact match
  if (isQIntType(a) && a == b) {
    return a;
  }

  if (isQIntType(a) || isQIntType(b)) {
    AT_ERROR(
        "promoteTypes with quantized numbers is not handled yet; figure out what the correct rules should be");
  }

  // this matrix has to be consistent with AT_FORALL_SCALAR_TYPES_WITH_COMPLEX
  // so that's why we have to add undefined as we are not sure what is the
  // corrent values for the type promotions in complex type cases.
  static constexpr ScalarType _promoteTypesLookup[static_cast<int>(
      ScalarType::NumOptions)][static_cast<int>(ScalarType::NumOptions)] = {
        /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  q1  q2  q3  bf*/
        /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, ud, ud, ud, u1, ud, ud, ud, ud},
        /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, ud, ud, ud, i1, ud, ud, ud, ud},
        /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, ud, ud, ud, i2, ud, ud, ud, ud},
        /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, ud, ud, ud, i4, ud, ud, ud, ud},
        /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, ud, ud, ud, i8, ud, ud, ud, ud},
        /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, ud, ud, ud, f2, ud, ud, ud, ud},
        /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, ud, ud, ud, f4, ud, ud, ud, ud},
        /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, ud, ud, ud, f8, ud, ud, ud, ud},
        /* c2 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* c4 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* c8 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, ud, ud, ud, b1, ud, ud, ud, ud},
        /* q1 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* q1 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* q2 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
        /* bf */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, bf},
  };
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

} // namespace c10
