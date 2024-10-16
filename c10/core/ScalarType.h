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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>
#include <unordered_map>

namespace c10 {

// dummy struct for uint1 to uint7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
template <unsigned int N>
struct dummy_uint1_7_t {};

// For the macros below:
//
// For users: If you want to macro some code for all non-QInt scalar types
// (i.e. types with complete information, you probably want one of the
// AT_FORALL_SCALAR_TYPES / AT_FORALL_SCALAR_TYPES_AND macros below, which are
// designed to behave similarly to the Dispatch macros with the same name.
//
// For adding a new dtype: In the beginning, we had an idea that there was a
// list of all scalar types, and you could use AT_FORALL_SCALAR_TYPES to
// iterate over them.  But over the years we added weird types which couldn't
// be handled uniformly everywhere and so in the end we ended up with some
// mish-mosh of some helper macros, but mostly use sites making a call about
// what dtypes they can or can't support.  So if you want to add a new dtype,
// the preferred resolution is to find a dtype similar to what you want,
// grep for it and edit all the sites you find this way.  If you need to add
// a completely new kind of dtype, you're going to have to laboriously audit
// all of the sites everywhere to figure out how it should work.  Consulting
// some old PRs where we added new dtypes (check history of this file) can
// help give you an idea where to start.

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
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz) /* 26 */      \
  _(uint16_t, UInt16) /* 27 */                           \
  _(uint32_t, UInt32) /* 28 */                           \
  _(uint64_t, UInt64) /* 29 */                           \
  _(c10::dummy_uint1_7_t<1>, UInt1) /* 30 */             \
  _(c10::dummy_uint1_7_t<2>, UInt2) /* 31 */             \
  _(c10::dummy_uint1_7_t<3>, UInt3) /* 32 */             \
  _(c10::dummy_uint1_7_t<4>, UInt4) /* 33 */             \
  _(c10::dummy_uint1_7_t<5>, UInt5) /* 34 */             \
  _(c10::dummy_uint1_7_t<6>, UInt6) /* 35 */             \
  _(c10::dummy_uint1_7_t<7>, UInt7) /* 36 */

// If you want to support ComplexHalf for real, add ComplexHalf
// into this macro (and change the name).  But beware: convert()
// doesn't work for all the conversions you need...
//
// TODO: To add unsigned int types here, we must define accumulate type.
// But uint8 currently accumulates into int64, so we would have to make
// an inconsistent choice for the larger types.  Difficult.
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

// This macro controls many of our C++ APIs, including constructors
// for Scalar as well as the data() and item() accessors on Tensor
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

// NB: despite its generic sounding name, the macros that don't take _AND
// are mostly only used by tensorexpr
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

// These macros are often controlling how many template instantiations we
// create for kernels.  It is typically inappropriate to add new dtypes here,
// instead, new types should be added to use sites on a case-by-case basis.
// We generally are not accepting new dtypes due to binary size concerns.

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

// NOLINTNEXTLINE(clang-diagnostic-unused-const-variable)
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

inline const char* toString(ScalarType t) {
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

inline size_t elementSize(ScalarType t) {
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

inline bool isIntegralType(ScalarType t, bool includeBool) {
  bool isIntegral =
      (t == ScalarType::Byte || t == ScalarType::Char || t == ScalarType::Int ||
       t == ScalarType::Long || t == ScalarType::Short ||
       t == ScalarType::UInt16 || t == ScalarType::UInt32 ||
       t == ScalarType::UInt64);

  return isIntegral || (includeBool && t == ScalarType::Bool);
}

C10_DEPRECATED_MESSAGE(
    "isIntegralType is deprecated. Please use the overload with 'includeBool' parameter instead.")
inline bool isIntegralType(ScalarType t) {
  return isIntegralType(t, /*includeBool=*/false);
}

inline bool isFloat8Type(ScalarType t) {
  return t == ScalarType::Float8_e5m2 || t == ScalarType::Float8_e5m2fnuz ||
      t == ScalarType::Float8_e4m3fn || t == ScalarType::Float8_e4m3fnuz;
}

inline bool isReducedFloatingType(ScalarType t) {
  return t == ScalarType::Half || t == ScalarType::BFloat16 || isFloat8Type(t);
}

inline bool isFloatingType(ScalarType t) {
  return t == ScalarType::Double || t == ScalarType::Float ||
      isReducedFloatingType(t);
}

inline bool isComplexType(ScalarType t) {
  return (
      t == ScalarType::ComplexHalf || t == ScalarType::ComplexFloat ||
      t == ScalarType::ComplexDouble);
}

inline bool isQIntType(ScalarType t) {
  // Don't forget to extend this when adding new QInt types
  return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
      t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
      t == ScalarType::QUInt2x4;
}

inline bool isBitsType(ScalarType t) {
  return t == ScalarType::Bits1x8 || t == ScalarType::Bits2x4 ||
      t == ScalarType::Bits4x2 || t == ScalarType::Bits8 ||
      t == ScalarType::Bits16;
}

inline bool isBarebonesUnsignedType(ScalarType t) {
  return t == ScalarType::UInt1 || t == ScalarType::UInt2 ||
      t == ScalarType::UInt3 || t == ScalarType::UInt4 ||
      t == ScalarType::UInt5 || t == ScalarType::UInt6 ||
      t == ScalarType::UInt7 || t == ScalarType::UInt16 ||
      t == ScalarType::UInt32 || t == ScalarType::UInt64;
}

inline ScalarType toQIntType(ScalarType t) {
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

inline ScalarType toUnderlying(ScalarType t) {
  switch (t) {
    case ScalarType::QUInt8:
    case ScalarType::QUInt4x2:
      [[fallthrough]];
    case ScalarType::QUInt2x4:
      return ScalarType::Byte;
    case ScalarType::QInt8:
      return ScalarType::Char;
    case ScalarType::QInt32:
      return ScalarType::Int;
    default:
      return t;
  }
}

inline bool isSignedType(ScalarType t) {
#define CASE_ISSIGNED(name)     \
  case ScalarType::name:        \
    return std::numeric_limits< \
        ::c10::impl::ScalarTypeToCPPTypeT<ScalarType::name>>::is_signed;

  switch (t) {
    case ScalarType::QInt8:
    case ScalarType::QUInt8:
    case ScalarType::QInt32:
    case ScalarType::QUInt4x2:
    case ScalarType::QUInt2x4:
      TORCH_CHECK(false, "isSignedType not supported for quantized types");
    case ScalarType::Bits1x8:
    case ScalarType::Bits2x4:
    case ScalarType::Bits4x2:
    case ScalarType::Bits8:
    case ScalarType::Bits16:
      TORCH_CHECK(false, "Bits types are undefined");
      CASE_ISSIGNED(UInt16);
      CASE_ISSIGNED(UInt32);
      CASE_ISSIGNED(UInt64);
      CASE_ISSIGNED(BFloat16);
      CASE_ISSIGNED(Float8_e5m2);
      CASE_ISSIGNED(Float8_e5m2fnuz);
      CASE_ISSIGNED(Float8_e4m3fn);
      CASE_ISSIGNED(Float8_e4m3fnuz);
      CASE_ISSIGNED(Byte);
      CASE_ISSIGNED(Char);
      CASE_ISSIGNED(Short);
      CASE_ISSIGNED(Int);
      CASE_ISSIGNED(Long);
      CASE_ISSIGNED(Half);
      CASE_ISSIGNED(Float);
      CASE_ISSIGNED(Double);
      CASE_ISSIGNED(ComplexHalf);
      CASE_ISSIGNED(ComplexFloat);
      CASE_ISSIGNED(ComplexDouble);
      CASE_ISSIGNED(Bool);
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
      return true;
    case ScalarType::Undefined:
    case ScalarType::NumOptions:
      break;
      // Do not add default here, but rather define behavior of every new entry
      // here.  `-Wswitch-enum` would raise a warning in those cases.
  }
  TORCH_CHECK(false, "Unknown ScalarType ", t);
#undef CASE_ISSIGNED
}

inline bool isUnderlying(ScalarType type, ScalarType qtype) {
  return type == toUnderlying(qtype);
}

inline ScalarType toRealValueType(ScalarType t) {
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

inline ScalarType toComplexType(ScalarType t) {
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
inline bool canCast(const ScalarType from, const ScalarType to) {
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

C10_API ScalarType promoteTypes(ScalarType a, ScalarType b);

inline std::ostream& operator<<(
    std::ostream& stream,
    at::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

// Returns a pair of strings representing the names for each dtype.
// The returned pair is (name, legacy_name_if_applicable)
C10_API std::pair<std::string, std::string> getDtypeNames(
    c10::ScalarType scalarType);

// Returns a map of string name to dtype.
C10_API const std::unordered_map<std::string, ScalarType>& getStringToDtypeMap();

} // namespace c10
