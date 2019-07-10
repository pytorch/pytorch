#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Half.h>
#include <c10/util/Optional.h>
#include <c10/util/typeid.h>

#include <complex>
#include <cstdint>
#include <iostream>

namespace c10 {

// TODO: check all usages of these macro and make sure
// the use case makes sense for qint

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_)       \
  _(uint8_t, Byte, i) /* 0 */                        \
  _(int8_t, Char, i) /* 1 */                         \
  _(int16_t, Short, i) /* 2 */                       \
  _(int, Int, i) /* 3 */                             \
  _(int64_t, Long, i) /* 4 */                        \
  _(at::Half, Half, d) /* 5 */                       \
  _(float, Float, d) /* 6 */                         \
  _(double, Double, d) /* 7 */                       \
  _(at::ComplexHalf, ComplexHalf, z) /* 8 */         \
  _(std::complex<float>, ComplexFloat, z) /* 9 */    \
  _(std::complex<double>, ComplexDouble, z) /* 10 */ \
  _(bool, Bool, i) /* 11 */                          \
  _(c10::qint8, QInt8, i) /* 12 */                   \
  _(c10::quint8, QUInt8, i) /* 13 */                 \
  _(c10::qint32, QInt32, i) /* 14 */                 \
  _(at::BFloat16, BFloat16, d) /* 15 */

// If you want to support ComplexHalf for real, replace occurrences
// of this macro with AT_FORALL_SCALAR_TYPES_WITH_COMPLEX.  But
// beware: convert() doesn't work for all the conversions you need...
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF(_) \
  _(uint8_t, Byte, i)                                              \
  _(int8_t, Char, i)                                               \
  _(int16_t, Short, i)                                             \
  _(int, Int, i)                                                   \
  _(int64_t, Long, i)                                              \
  _(at::Half, Half, d)                                             \
  _(float, Float, d)                                               \
  _(double, Double, d)                                             \
  _(std::complex<float>, ComplexFloat, z)                          \
  _(std::complex<double>, ComplexDouble, z)                        \
  _(bool, Bool, i)                                                 \
  _(c10::qint8, QInt8, i)                                          \
  _(c10::quint8, QUInt8, i)                                        \
  _(c10::qint32, QInt32, i)                                        \
  _(at::BFloat16, BFloat16, d)

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_AND_QINT(_) \
  _(uint8_t, Byte, i)                                                       \
  _(int8_t, Char, i)                                                        \
  _(int16_t, Short, i)                                                      \
  _(int, Int, i)                                                            \
  _(int64_t, Long, i)                                                       \
  _(at::Half, Half, d)                                                      \
  _(float, Float, d)                                                        \
  _(double, Double, d)                                                      \
  _(std::complex<float>, ComplexFloat, z)                                   \
  _(std::complex<double>, ComplexDouble, z)                                 \
  _(bool, Bool, i)                                                          \
  _(at::BFloat16, BFloat16, d)

#define AT_FORALL_SCALAR_TYPES_EXCEPT_QINT(_) \
  _(uint8_t, Byte, i)                         \
  _(int8_t, Char, i)                          \
  _(int16_t, Short, i)                        \
  _(int, Int, i)                              \
  _(int64_t, Long, i)                         \
  _(at::Half, Half, d)                        \
  _(float, Float, d)                          \
  _(double, Double, d)                        \
  _(at::BFloat16, BFloat16, d)

#define AT_FORALL_SCALAR_TYPES_AND_BOOL_EXCEPT_QINT(_) \
  _(uint8_t, Byte, i)                                  \
  _(int8_t, Char, i)                                   \
  _(int16_t, Short, i)                                 \
  _(int, Int, i)                                       \
  _(int64_t, Long, i)                                  \
  _(at::Half, Half, d)                                 \
  _(float, Float, d)                                   \
  _(double, Double, d)                                 \
  _(bool, Bool, i)

#define AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(_) \
  _(uint8_t, Byte, i)                         \
  _(int8_t, Char, i)                          \
  _(int16_t, Short, i)                        \
  _(int, Int, i)                              \
  _(int64_t, Long, i)                         \
  _(float, Float, d)                          \
  _(double, Double, d)                        \
  _(c10::qint8, QInt8, i)                     \
  _(c10::quint8, QUInt8, i)                   \
  _(c10::qint32, QInt32, i)

#define AT_FORALL_SCALAR_TYPES_EXCEPT_HALF_AND_QINT(_) \
  _(uint8_t, Byte, i)                                  \
  _(int8_t, Char, i)                                   \
  _(int16_t, Short, i)                                 \
  _(int, Int, i)                                       \
  _(int64_t, Long, i)                                  \
  _(float, Float, d)                                   \
  _(double, Double, d)

enum class ScalarType : int8_t {
#define DEFINE_ENUM(_1, n, _2) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_ENUM)
#undef DEFINE_ENUM
      Undefined,
  NumOptions
};

static inline caffe2::TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
#define DEFINE_CASE(ctype, name, _) \
  case ScalarType::name:            \
    return caffe2::TypeMeta::Make<ctype>();

  switch (scalar_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
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
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_IF)
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

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name, _2) \
  case ScalarType::name:         \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CASE)
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
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CASE_ELEMENTSIZE_CASE)
    default:
      AT_ERROR("Unknown ScalarType");
  }
#undef CASE_ELEMENTSIZE_CASE
}

static inline bool isIntegralType(ScalarType t) {
  return (
      t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
      t == ScalarType::Long || t == ScalarType::Short);
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
      /* u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1 */
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, ud, ud, ud, u1},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, ud, ud, ud, i1},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, ud, ud, ud, i2},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, ud, ud, ud, i4},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, ud, ud, ud, i8},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, ud, ud, ud, f2},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, ud, ud, ud, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, ud, ud, ud, f8},
      /* c2 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* c4 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* c8 */ {ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud, ud},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, ud, ud, ud, b1},
  };
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

} // namespace c10
