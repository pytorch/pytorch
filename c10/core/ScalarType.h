#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Half.h>
#include <c10/util/bits.h>
#include <c10/util/complex.h>
#include <c10/util/qint32.h>
#include <c10/util/qint8.h>
#include <c10/util/quint2x4.h>
#include <c10/util/quint4x2.h>
#include <c10/util/quint8.h>

#include <array>
#include <complex>
#include <cstdint>
#include <ostream>

namespace c10 {

// For the macros below:
// NB: If you want to macro some code for all non-QInt scalar types (i.e. types
// with complete information, you probably want one of the
// AT_FORALL_SCALAR_TYPES / AT_FORALL_SCALAR_TYPES_AND
// macros below, which are designed to behave similarly to the Dispatch macros
// with the same name.

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(at::Half, Half) /* 5 */                              \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \
  _(c10::complex<float>, ComplexFloat) /* 9 */           \
  _(c10::complex<double>, ComplexDouble) /* 10 */        \
  _(bool, Bool) /* 11 */                                 \
  _(c10::qint8, QInt8) /* 12 */                          \
  _(c10::quint8, QUInt8) /* 13 */                        \
  _(c10::qint32, QInt32) /* 14 */                        \
  _(at::BFloat16, BFloat16) /* 15 */                     \
  _(c10::quint4x2, QUInt4x2) /* 16 */                    \
  _(c10::quint2x4, QUInt2x4) /* 17 */                    \
  _(c10::bits1x8, Bits1x8) /* 18 */                      \
  _(c10::bits2x4, Bits2x4) /* 19 */                      \
  _(c10::bits4x2, Bits4x2) /* 20 */                      \
  _(c10::bits8, Bits8) /* 21 */                          \
  _(c10::bits16, Bits16) /* 22 */                        \
  _(c10::Float8_e5m2, Float8_e5m2) /* 23 */              \
  _(c10::Float8_e4m3fn, Float8_e4m3fn) /* 24 */          \
  _(c10::Float8_e5m2fnuz, Float8_e5m2fnuz) /* 25 */      \
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz) /* 26 */

// If you want to support ComplexHalf for real, add ComplexHalf
// into this macro (and change the name).  But beware: convert()
// doesn't work for all the conversions you need...
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(_) \
  _(uint8_t, Byte)                                                      \
  _(int8_t, Char)                                                       \
  _(int16_t, Short)                                                     \
  _(int, Int)                                                           \
  _(int64_t, Long)                                                      \
  _(at::Half, Half)                                                     \
  _(float, Float)                                                       \
  _(double, Double)                                                     \
  _(c10::complex<float>, ComplexFloat)                                  \
  _(c10::complex<double>, ComplexDouble)                                \
  _(bool, Bool)                                                         \
  _(at::BFloat16, BFloat16)                                             \
  _(at::Float8_e5m2, Float8_e5m2)                                       \
  _(at::Float8_e4m3fn, Float8_e4m3fn)

#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
  _(uint8_t, Byte)                             \
  _(int8_t, Char)                              \
  _(int16_t, Short)                            \
  _(int, Int)                                  \
  _(int64_t, Long)                             \
  _(at::Half, Half)                            \
  _(float, Float)                              \
  _(double, Double)                            \
  _(c10::complex<c10::Half>, ComplexHalf)      \
  _(c10::complex<float>, ComplexFloat)         \
  _(c10::complex<double>, ComplexDouble)       \
  _(bool, Bool)                                \
  _(at::BFloat16, BFloat16)                    \
  _(at::Float8_e5m2, Float8_e5m2)              \
  _(at::Float8_e4m3fn, Float8_e4m3fn)          \
  _(at::Float8_e5m2fnuz, Float8_e5m2fnuz)      \
  _(at::Float8_e4m3fnuz, Float8_e4m3fnuz)

enum class ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};

constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

namespace impl {

// These are used to map ScalarTypes to C++ types.

template <c10::ScalarType N>
struct ScalarTypeToCPPType;

#define SPECIALIZE_ScalarTypeToCPPType(cpp_type, scalar_type)                \
  template <>                                                                \
  struct ScalarTypeToCPPType<c10::ScalarType::scalar_type> {                 \
    using type = cpp_type;                                                   \
                                                                             \
    /* This is a workaround for the CUDA bug which prevents */               \
    /* ::detail::ScalarTypeToCType<T>::type being used directly due to */    \
    /* ambiguous reference which can't to be resolved. For some reason it */ \
    /* can't pick between at::detail and at::cuda::detail. */                \
    /* For repro example, please see: */                                     \
    /* https://gist.github.com/izdeby/952ae7cf256ddb740a73776d39a7e7ba */    \
    /* TODO: remove once the bug is fixed. */                                \
    static type t;                                                           \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

} // namespace impl

template <typename T>
struct CppTypeToScalarType;

#define SPECIALIZE_CppTypeToScalarType(cpp_type, scalar_type)                  \
  template <>                                                                  \
  struct CppTypeToScalarType<cpp_type>                                         \
      : std::                                                                  \
            integral_constant<c10::ScalarType, c10::ScalarType::scalar_type> { \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_CppTypeToScalarType)

#undef SPECIALIZE_CppTypeToScalarType

#define AT_FORALL_INT_TYPES(_) \
  _(uint8_t, Byte)             \
  _(int8_t, Char)              \
  _(int16_t, Short)            \
  _(int, Int)                  \
  _(int64_t, Long)

#define AT_FORALL_SCALAR_TYPES(_) \
  _(uint8_t, Byte)                \
  _(int8_t, Char)                 \
  _(int16_t, Short)               \
  _(int, Int)                     \
  _(int64_t, Long)                \
  _(float, Float)                 \
  _(double, Double)

#define AT_FORALL_SCALAR_TYPES_AND(SCALARTYPE, _) \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE>::t),  \
    SCALARTYPE)

#define AT_FORALL_SCALAR_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _) \
  _(uint8_t, Byte)                                               \
  _(int8_t, Char)                                                \
  _(int16_t, Short)                                              \
  _(int, Int)                                                    \
  _(int64_t, Long)                                               \
  _(float, Float)                                                \
  _(double, Double)                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE1>::t),                \
    SCALARTYPE1)                                                 \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                   \
             ::c10::ScalarType::SCALARTYPE2>::t),                \
    SCALARTYPE2)

#define AT_FORALL_SCALAR_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, _) \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE1>::t),                             \
    SCALARTYPE1)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE2>::t),                             \
    SCALARTYPE2)                                                              \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                                \
             ::c10::ScalarType::SCALARTYPE3>::t),                             \
    SCALARTYPE3)

#define AT_FORALL_SCALAR_TYPES_AND4(                       \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, _) \
  _(uint8_t, Byte)                                         \
  _(int8_t, Char)                                          \
  _(int16_t, Short)                                        \
  _(int, Int)                                              \
  _(int64_t, Long)                                         \
  _(float, Float)                                          \
  _(double, Double)                                        \
  _(decltype(::c10::impl::ScalarTypeToCPPType<             \
             ::c10::ScalarType::SCALARTYPE1>::t),          \
    SCALARTYPE1)                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<             \
             ::c10::ScalarType::SCALARTYPE2>::t),          \
    SCALARTYPE2)                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<             \
             ::c10::ScalarType::SCALARTYPE3>::t),          \
    SCALARTYPE3)                                           \
  _(decltype(::c10::impl::ScalarTypeToCPPType<             \
             ::c10::ScalarType::SCALARTYPE4>::t),          \
    SCALARTYPE4)

#define AT_FORALL_SCALAR_TYPES_AND5(                                    \
    SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, SCALARTYPE4, SCALARTYPE5, _) \
  _(uint8_t, Byte)                                                      \
  _(int8_t, Char)                                                       \
  _(int16_t, Short)                                                     \
  _(int, Int)                                                           \
  _(int64_t, Long)                                                      \
  _(float, Float)                                                       \
  _(double, Double)                                                     \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                          \
             ::c10::ScalarType::SCALARTYPE1>::t),                       \
    SCALARTYPE1)                                                        \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                          \
             ::c10::ScalarType::SCALARTYPE2>::t),                       \
    SCALARTYPE2)                                                        \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                          \
             ::c10::ScalarType::SCALARTYPE3>::t),                       \
    SCALARTYPE3)                                                        \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                          \
             ::c10::ScalarType::SCALARTYPE4>::t),                       \
    SCALARTYPE4)                                                        \
  _(decltype(::c10::impl::ScalarTypeToCPPType<                          \
             ::c10::ScalarType::SCALARTYPE5>::t),                       \
    SCALARTYPE5)

#define AT_FORALL_SCALAR_TYPES_AND6(              \
    SCALARTYPE1,                                  \
    SCALARTYPE2,                                  \
    SCALARTYPE3,                                  \
    SCALARTYPE4,                                  \
    SCALARTYPE5,                                  \
    SCALARTYPE6,                                  \
    _)                                            \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE3>::t), \
    SCALARTYPE3)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE4>::t), \
    SCALARTYPE4)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE5>::t), \
    SCALARTYPE5)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE6>::t), \
    SCALARTYPE6)

#define AT_FORALL_SCALAR_TYPES_AND7(              \
    SCALARTYPE1,                                  \
    SCALARTYPE2,                                  \
    SCALARTYPE3,                                  \
    SCALARTYPE4,                                  \
    SCALARTYPE5,                                  \
    SCALARTYPE6,                                  \
    SCALARTYPE7,                                  \
    _)                                            \
  _(uint8_t, Byte)                                \
  _(int8_t, Char)                                 \
  _(int16_t, Short)                               \
  _(int, Int)                                     \
  _(int64_t, Long)                                \
  _(float, Float)                                 \
  _(double, Double)                               \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE1>::t), \
    SCALARTYPE1)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE2>::t), \
    SCALARTYPE2)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE3>::t), \
    SCALARTYPE3)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE4>::t), \
    SCALARTYPE4)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE5>::t), \
    SCALARTYPE5)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE6>::t), \
    SCALARTYPE6)                                  \
  _(decltype(::c10::impl::ScalarTypeToCPPType<    \
             ::c10::ScalarType::SCALARTYPE7>::t), \
    SCALARTYPE7)

#define AT_FORALL_QINT_TYPES(_) \
  _(c10::qint8, QInt8)          \
  _(c10::quint8, QUInt8)        \
  _(c10::qint32, QInt32)        \
  _(c10::quint4x2, QUInt4x2)    \
  _(c10::quint2x4, QUInt2x4)

#define AT_FORALL_COMPLEX_TYPES(_)     \
  _(c10::complex<float>, ComplexFloat) \
  _(c10::complex<double>, ComplexDouble)

#define DEFINE_CONSTANT(_, name) \
  constexpr ScalarType k##name = ScalarType::name;

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

static inline const char* toString(ScalarType t) {
#define DEFINE_CASE(_, name) \
  case ScalarType::name:     \
    return #name;

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
    default:
      return "UNKNOWN_SCALAR";
  }
#undef DEFINE_CASE
}

static inline size_t elementSize(ScalarType t) {
#define CASE_ELEMENTSIZE_CASE(ctype, name) \
  case ScalarType::name:                   \
    return sizeof(ctype);

  switch (t) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(CASE_ELEMENTSIZE_CASE)
    default:
      TORCH_CHECK(false, "Unknown ScalarType");
  }
#undef CASE_ELEMENTSIZE_CASE
}

static inline bool isIntegralType(ScalarType t, bool includeBool) {
  bool isIntegral =
      (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
       t == ScalarType::Long || t == ScalarType::Short);

  return isIntegral || (includeBool && t == ScalarType::Bool);
}

C10_DEPRECATED_MESSAGE(
    "isIntegralType is deprecated. Please use the overload with 'includeBool' parameter instead.")
static inline bool isIntegralType(ScalarType t) {
  return isIntegralType(t, /*includeBool=*/false);
}

static inline bool isFloat8Type(ScalarType t) {
  return t == ScalarType::Float8_e5m2 || t == ScalarType::Float8_e5m2fnuz ||
      t == ScalarType::Float8_e4m3fn || t == ScalarType::Float8_e4m3fnuz;
}

static inline bool isReducedFloatingType(ScalarType t) {
  return t == ScalarType::Half || t == ScalarType::BFloat16 || isFloat8Type(t);
}

static inline bool isFloatingType(ScalarType t) {
  return t == ScalarType::Double || t == ScalarType::Float ||
      isReducedFloatingType(t);
}

static inline bool isComplexType(ScalarType t) {
  return (
      t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat ||
      t == ScalarType::ComplexDouble);
}

static inline bool isQIntType(ScalarType t) {
  // Don't forget to extend this when adding new QInt types
  return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
      t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
      t == ScalarType::QUInt2x4;
}

static inline bool isBitsType(ScalarType t) {
  return t == ScalarType::Bits1x8 || t == ScalarType::Bits2x4 ||
      t == ScalarType::Bits4x2 || t == ScalarType::Bits8 ||
      t == ScalarType::Bits16;
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
    case ScalarType::QUInt4x2:
      return ScalarType::Byte;
    case ScalarType::QUInt2x4:
      return ScalarType::Byte;
    default:
      return t;
  }
}

static inline bool isSignedType(ScalarType t) {
  TORCH_CHECK(!isQIntType(t), "isSignedType not supported for quantized types");
#define CASE_SIGNED(ctype, name) \
  case ScalarType::name:         \
    return std::numeric_limits<ctype>::is_signed;

  switch (t) {
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      TORCH_CHECK(false, "Bits types are undefined");
    case ScalarType::ComplexHalf:
    case ScalarType::ComplexFloat:
    case ScalarType::ComplexDouble:
      return true;
      AT_FORALL_SCALAR_TYPES_AND5(
          Half, Bool, BFloat16, Float8_e5m2, Float8_e4m3fn, CASE_SIGNED)
    default:
      TORCH_CHECK(false, "Unknown ScalarType");
  }
#undef CASE_SIGNED
}

static inline bool isUnderlying(ScalarType type, ScalarType qtype) {
  return type == toUnderlying(qtype);
}

static inline ScalarType toRealValueType(ScalarType t) {
  switch (t) {
    case ScalarType::ComplexHalf:
      return ScalarType::Half;
    case ScalarType::ComplexFloat:
      return ScalarType::Float;
    case ScalarType::ComplexDouble:
      return ScalarType::Double;
    default:
      return t;
  }
}

static inline ScalarType toComplexType(ScalarType t) {
  switch (t) {
    case ScalarType::BFloat16:
      // BFloat16 has range equivalent to Float,
      // so we map it to ComplexFloat.
      return ScalarType::ComplexFloat;
    case ScalarType::Half:
      return ScalarType::ComplexHalf;
    case ScalarType::Float:
      return ScalarType::ComplexFloat;
    case ScalarType::Double:
      return ScalarType::ComplexDouble;
    case ScalarType::ComplexHalf:
      return ScalarType::ComplexHalf;
    case ScalarType::ComplexFloat:
      return ScalarType::ComplexFloat;
    case ScalarType::ComplexDouble:
      return ScalarType::ComplexDouble;
    default:
      TORCH_CHECK(false, "Unknown Complex ScalarType for ", t);
  }
}

// see tensor_attributes.rst for detailed explanation and examples
// of casting rules.
static inline bool canCast(const ScalarType from, const ScalarType to) {
  // We disallow complex -> non complex, e.g., float_tensor *= complex is
  // disallowed.
  if (isComplexType(from) && !isComplexType(to)) {
    return false;
  }
  // We disallow float -> integral, e.g., int_tensor *= float is disallowed.
  if (isFloatingType(from) && isIntegralType(to, false)) {
    return false;
  }

  // Treat bool as a distinct "category," to be consistent with type promotion
  // rules (e.g. `bool_tensor + 5 -> int64_tensor`). If `5` was in the same
  // category as `bool_tensor`, we would not promote. Differing categories
  // implies `bool_tensor += 5` is disallowed.
  //
  // NB: numpy distinguishes "unsigned" as a category to get the desired
  // `bool_tensor + 5 -> int64_tensor` behavior. We don't, because:
  // * We don't want the performance hit of checking the runtime sign of
  // Scalars.
  // * `uint8_tensor + 5 -> int64_tensor` would be undesirable.
  if (from != ScalarType::Bool && to == ScalarType::Bool) {
    return false;
  }
  return true;
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
  constexpr auto c2 = ScalarType::ComplexHalf;
  constexpr auto c4 = ScalarType::ComplexFloat;
  constexpr auto c8 = ScalarType::ComplexDouble;
  constexpr auto b1 = ScalarType::Bool;
  constexpr auto bf = ScalarType::BFloat16;
  constexpr auto ud = ScalarType::Undefined;
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }

  // If the two types are equal, return that type
  if (a == b) {
    return a;
  }

  // Handle identically equal types
  if (isQIntType(a) || isQIntType(b)) {
    TORCH_CHECK(
        false,
        "promoteTypes with quantized numbers is not handled yet; figure out what the correct rules should be, offending types: ",
        toString(a),
        " ",
        toString(b));
  }

  if (isBitsType(a) || isBitsType(b)) {
    return ScalarType::Undefined;
  }

  if (isFloat8Type(a) || isFloat8Type(b)) {
    TORCH_CHECK(
        false,
        "Promotion for Float8 Types is not supported, attempted to promote ",
        toString(a),
        " and ",
        toString(b));
  }

  // Bits, Quantized and Float8 are 14 dtypes already handled and not included
  // in the promotion table below.
  static constexpr int num_bits_types = static_cast<int>(ScalarType::Bits16) -
      static_cast<int>(ScalarType::Bits1x8) + 1;

  static constexpr int num_float8_types =
      static_cast<int>(ScalarType::Float8_e4m3fnuz) -
      static_cast<int>(ScalarType::Float8_e5m2) + 1;

  static constexpr int num_qint_types = static_cast<int>(ScalarType::QInt32) -
      static_cast<int>(ScalarType::QInt8) + 1;

  static constexpr int num_quint_types =
      static_cast<int>(ScalarType::QUInt2x4) -
      static_cast<int>(ScalarType::QUInt4x2) + 1;

  static constexpr int num_quantized_types = num_qint_types + num_quint_types;

  static constexpr int num_missing_types =
      num_bits_types + num_float8_types + num_quantized_types;

  // Bfloat16 is at position 15 in the ScalerType enum, There are three types
  // below bf16 not included in the table, Qint8, QUInt8, QInt32. Every other
  // type above bf16, i.e. {Bits, Quantized, Float8} are not included in the
  // table.

  // If either of the types is bf16, we need to shift the type down by the one
  // missing section in the table that is less then bf16 i.e {QInt8, QUInt8,
  // QInt32}
  a = a == bf ? static_cast<ScalarType>(static_cast<int>(a) - num_qint_types)
              : a;
  b = b == bf ? static_cast<ScalarType>(static_cast<int>(b) - num_qint_types)
              : b;

  // We decrease the promotion table by the number of missing types -> 14
  // and then subtract 1 more from the table since we don't store ud to ud
  // mapping.
  static constexpr int NUM_PROMOTE_TYPES =
      static_cast<int>(ScalarType::NumOptions) - num_missing_types - 1;

  // this matrix has to be consistent with
  // AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS undefined is used where we
  // are not sure about the correct value for type promotion.
  // clang-format off
  static constexpr std::
  array<std::array<ScalarType, NUM_PROMOTE_TYPES>, NUM_PROMOTE_TYPES>
      _promoteTypesLookup = {{
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  bf*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1, bf},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1, bf},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2, bf},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4, bf},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8, bf},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8, f8},
      /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2, c4},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c4, c8, bf, bf},
  }};
  // clang-format on
  return _promoteTypesLookup[static_cast<int>(a)][static_cast<int>(b)];
}

inline std::ostream& operator<<(
    std::ostream& stream,
    at::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

} // namespace c10
