#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Exception.h>
#include <c10/util/Float4_e2m1fn_x2.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e5m2.h>
#include <c10/util/Float8_e5m2fnuz.h>
#include <c10/util/Float8_e8m0fnu.h>
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
#include <limits>
#include <ostream>
#include <type_traits>
#include <unordered_map>

#include <torch/headeronly/core/ScalarType.h>

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-enum")
C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wswitch-default")

namespace c10 {

// See [dtype Macros note] in torch/headeronly/core/ScalarType.h
// regarding macros.

#define DEFINE_CONSTANT(_, name) \
  constexpr ScalarType k##name = ScalarType::name;

// NOLINTNEXTLINE(clang-diagnostic-unused-const-variable)
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CONSTANT)
#undef DEFINE_CONSTANT

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

[[deprecated(
    "isIntegralType is deprecated. Please use the overload with 'includeBool' parameter instead.")]] inline bool
isIntegralType(ScalarType t) {
  return isIntegralType(t, /*includeBool=*/false);
}

inline bool isFloat8Type(ScalarType t) {
  return t == ScalarType::Float8_e5m2 || t == ScalarType::Float8_e5m2fnuz ||
      t == ScalarType::Float8_e4m3fn || t == ScalarType::Float8_e4m3fnuz ||
      t == ScalarType::Float8_e8m0fnu;
}

inline bool isReducedFloatingType(ScalarType t) {
  return t == ScalarType::Half || t == ScalarType::BFloat16 ||
      isFloat8Type(t) || t == ScalarType::Float4_e2m1fn_x2;
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

inline bool isSignedType(ScalarType t) {
#define CASE_ISSIGNED(name)     \
  case ScalarType::name:        \
    return std::numeric_limits< \
        ::c10::impl::ScalarTypeToCPPTypeT<ScalarType::name>>::is_signed;

  // TODO(#146647): If we expect to have numeric_limits for everything,
  // let's just have a big macro for the whole thing.
  // If we're hardcoding it, let's just use the macro and a "true"/"false"
  // below?
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
      CASE_ISSIGNED(Float8_e8m0fnu);
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
    case ScalarType::Int1:
    case ScalarType::Int2:
    case ScalarType::Int3:
    case ScalarType::Int4:
    case ScalarType::Int5:
    case ScalarType::Int6:
    case ScalarType::Int7:
    case ScalarType::Float4_e2m1fn_x2:
      return true;
    case ScalarType::UInt1:
    case ScalarType::UInt2:
    case ScalarType::UInt3:
    case ScalarType::UInt4:
    case ScalarType::UInt5:
    case ScalarType::UInt6:
    case ScalarType::UInt7:
      return false;
    case ScalarType::Undefined:
    case ScalarType::NumOptions:
      break;
      // Do not add default here, but rather define behavior of every new entry
      // here.  `-Wswitch-enum` would raise a warning in those cases.
      // TODO: get PyTorch to adopt exhaustive switches by default with a way to
      // opt specific switches to being non-exhaustive.
      // Exhaustive:
      // `-Wswitch-enum`, `-Wswitch-default`, `-Wno-covered-switch-default`
      // Non-Exhaustive:
      // `-Wno-switch-enum`, `-Wswitch-default`, `-Wcovered-switch-default`
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

// Returns a pair of strings representing the names for each dtype.
// The returned pair is (name, legacy_name_if_applicable)
C10_API std::pair<std::string, std::string> getDtypeNames(
    c10::ScalarType scalarType);

// Returns a map of string name to dtype.
C10_API const std::unordered_map<std::string, ScalarType>& getStringToDtypeMap();

} // namespace c10

C10_DIAGNOSTIC_POP()
