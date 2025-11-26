#pragma once

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float4_e2m1fn_x2.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Float8_e8m0fnu.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/bits.h>
#include <torch/headeronly/util/complex.h>
#include <torch/headeronly/util/qint32.h>
#include <torch/headeronly/util/qint8.h>
#include <torch/headeronly/util/quint2x4.h>
#include <torch/headeronly/util/quint4x2.h>
#include <torch/headeronly/util/quint8.h>

#include <cstdint>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")

namespace c10 {

// dummy struct for uint1 to uint7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
template <unsigned int N>
struct dummy_uint1_7_t {};

// dummy struct for int1 to int7, actual functionality
// of these dtypes will be implemented in python with Tensor subclass
template <unsigned int N>
struct dummy_int1_7_t {};

// [dtype Macros note] For the macros below:
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
  _(c10::Half, Half)                                                    \
  _(float, Float)                                                       \
  _(double, Double)                                                     \
  _(c10::complex<float>, ComplexFloat)                                  \
  _(c10::complex<double>, ComplexDouble)                                \
  _(bool, Bool)                                                         \
  _(c10::BFloat16, BFloat16)                                            \
  _(c10::Float8_e5m2, Float8_e5m2)                                      \
  _(c10::Float8_e4m3fn, Float8_e4m3fn)

// This macro controls many of our C++ APIs, including constructors
// for Scalar as well as the data() and item() accessors on Tensor
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(_) \
  _(uint8_t, Byte)                             \
  _(int8_t, Char)                              \
  _(int16_t, Short)                            \
  _(int, Int)                                  \
  _(int64_t, Long)                             \
  _(c10::Half, Half)                           \
  _(float, Float)                              \
  _(double, Double)                            \
  _(c10::complex<c10::Half>, ComplexHalf)      \
  _(c10::complex<float>, ComplexFloat)         \
  _(c10::complex<double>, ComplexDouble)       \
  _(bool, Bool)                                \
  _(c10::BFloat16, BFloat16)                   \
  _(c10::Float8_e5m2, Float8_e5m2)             \
  _(c10::Float8_e4m3fn, Float8_e4m3fn)         \
  _(c10::Float8_e5m2fnuz, Float8_e5m2fnuz)     \
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz)     \
  _(c10::Float8_e8m0fnu, Float8_e8m0fnu)

// NB: Order matters for this macro; it is relied upon in
// _promoteTypesLookup and the serialization format.
#define AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(_) \
  _(uint8_t, Byte) /* 0 */                               \
  _(int8_t, Char) /* 1 */                                \
  _(int16_t, Short) /* 2 */                              \
  _(int, Int) /* 3 */                                    \
  _(int64_t, Long) /* 4 */                               \
  _(c10::Half, Half) /* 5 */                             \
  _(float, Float) /* 6 */                                \
  _(double, Double) /* 7 */                              \
  _(c10::complex<c10::Half>, ComplexHalf) /* 8 */        \
  _(c10::complex<float>, ComplexFloat) /* 9 */           \
  _(c10::complex<double>, ComplexDouble) /* 10 */        \
  _(bool, Bool) /* 11 */                                 \
  _(c10::qint8, QInt8) /* 12 */                          \
  _(c10::quint8, QUInt8) /* 13 */                        \
  _(c10::qint32, QInt32) /* 14 */                        \
  _(c10::BFloat16, BFloat16) /* 15 */                    \
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
  _(c10::dummy_uint1_7_t<7>, UInt7) /* 36 */             \
  _(c10::dummy_int1_7_t<1>, Int1) /* 37 */               \
  _(c10::dummy_int1_7_t<2>, Int2) /* 38 */               \
  _(c10::dummy_int1_7_t<3>, Int3) /* 39 */               \
  _(c10::dummy_int1_7_t<4>, Int4) /* 40 */               \
  _(c10::dummy_int1_7_t<5>, Int5) /* 41 */               \
  _(c10::dummy_int1_7_t<6>, Int6) /* 42 */               \
  _(c10::dummy_int1_7_t<7>, Int7) /* 43 */               \
  _(c10::Float8_e8m0fnu, Float8_e8m0fnu) /* 44 */        \
  _(c10::Float4_e2m1fn_x2, Float4_e2m1fn_x2) /* 45 */

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
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE>, SCALARTYPE)

#define AT_FORALL_SCALAR_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, _)   \
  _(uint8_t, Byte)                                                 \
  _(int8_t, Char)                                                  \
  _(int16_t, Short)                                                \
  _(int, Int)                                                      \
  _(int64_t, Long)                                                 \
  _(float, Float)                                                  \
  _(double, Double)                                                \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE1>, \
    SCALARTYPE1)                                                   \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE2>, SCALARTYPE2)

#define AT_FORALL_SCALAR_TYPES_AND3(SCALARTYPE1, SCALARTYPE2, SCALARTYPE3, _) \
  _(uint8_t, Byte)                                                            \
  _(int8_t, Char)                                                             \
  _(int16_t, Short)                                                           \
  _(int, Int)                                                                 \
  _(int64_t, Long)                                                            \
  _(float, Float)                                                             \
  _(double, Double)                                                           \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE1>,            \
    SCALARTYPE1)                                                              \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE2>,            \
    SCALARTYPE2)                                                              \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE3>, SCALARTYPE3)

#define AT_FORALL_SCALAR_TYPES_AND7(                               \
    SCALARTYPE1,                                                   \
    SCALARTYPE2,                                                   \
    SCALARTYPE3,                                                   \
    SCALARTYPE4,                                                   \
    SCALARTYPE5,                                                   \
    SCALARTYPE6,                                                   \
    SCALARTYPE7,                                                   \
    _)                                                             \
  _(uint8_t, Byte)                                                 \
  _(int8_t, Char)                                                  \
  _(int16_t, Short)                                                \
  _(int, Int)                                                      \
  _(int64_t, Long)                                                 \
  _(float, Float)                                                  \
  _(double, Double)                                                \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE1>, \
    SCALARTYPE1)                                                   \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE2>, \
    SCALARTYPE2)                                                   \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE3>, \
    SCALARTYPE3)                                                   \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE4>, \
    SCALARTYPE4)                                                   \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE5>, \
    SCALARTYPE5)                                                   \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE6>, \
    SCALARTYPE6)                                                   \
  _(c10::impl::ScalarTypeToCPPTypeT<c10::ScalarType::SCALARTYPE7>, SCALARTYPE7)

#define AT_FORALL_QINT_TYPES(_) \
  _(c10::qint8, QInt8)          \
  _(c10::quint8, QUInt8)        \
  _(c10::qint32, QInt32)        \
  _(c10::quint4x2, QUInt4x2)    \
  _(c10::quint2x4, QUInt2x4)

#define AT_FORALL_FLOAT8_TYPES(_)          \
  _(c10::Float8_e5m2, Float8_e5m2)         \
  _(c10::Float8_e4m3fn, Float8_e4m3fn)     \
  _(c10::Float8_e5m2fnuz, Float8_e5m2fnuz) \
  _(c10::Float8_e4m3fnuz, Float8_e4m3fnuz) \
  _(c10::Float8_e8m0fnu, Float8_e8m0fnu)

#define AT_FORALL_COMPLEX_TYPES(_)     \
  _(c10::complex<float>, ComplexFloat) \
  _(c10::complex<double>, ComplexDouble)

enum class ScalarType : int8_t {
#define DEFINE_ST_ENUM_VAL_(_1, n) n,
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_ST_ENUM_VAL_)
#undef DEFINE_ENUM_ST_ENUM_VAL_
      Undefined,
  NumOptions
};

constexpr uint16_t NumScalarTypes =
    static_cast<uint16_t>(ScalarType::NumOptions);

// Map from C++ type to ScalarType enum
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
    /* UPDATE: while the CUDA bug is fixed, we cannot remove the  */         \
    /* workaround as it is BC breaking. However, it is recommended to  */    \
    /* update any code that contains */                                      \
    /*   decltype(ScalarTypeToCPPType<T>::t) */                              \
    /* with */                                                               \
    /*   ScalarTypeToCPPTypeT<T> */                                          \
    static type t;                                                           \
  };

AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(SPECIALIZE_ScalarTypeToCPPType)

#undef SPECIALIZE_ScalarTypeToCPPType

template <c10::ScalarType N>
using ScalarTypeToCPPTypeT = typename ScalarTypeToCPPType<N>::type;

} // namespace impl

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

inline std::ostream& operator<<(
    std::ostream& stream,
    c10::ScalarType scalar_type) {
  return stream << toString(scalar_type);
}

inline bool isQIntType(ScalarType t) {
  // Don't forget to extend this when adding new QInt types
  return t == ScalarType::QInt8 || t == ScalarType::QUInt8 ||
      t == ScalarType::QInt32 || t == ScalarType::QUInt4x2 ||
      t == ScalarType::QUInt2x4;
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

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::CppTypeToScalarType;
using c10::dummy_int1_7_t;
using c10::dummy_uint1_7_t;
using c10::NumScalarTypes;
using c10::ScalarType;
using c10::toString;
using c10::operator<<;
using c10::isQIntType;
using c10::toUnderlying;

namespace impl {
using c10::impl::ScalarTypeToCPPTypeT;
} // namespace impl

HIDDEN_NAMESPACE_END(torch, headeronly)

C10_DIAGNOSTIC_POP()
