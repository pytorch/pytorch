#pragma once

#include <string>

#include "lazy_tensors/primitive_types.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {
namespace primitive_util {

// Returns the primitive type (eg, F32) corresponding to the given
// template parameter native type (eg, float).
template <typename NativeT>
PrimitiveType NativeToPrimitiveType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only appears if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to primitive type.");
  return PrimitiveType::INVALID;
}

// Declarations of specializations for each native type which correspond to a
// primitive type.  As an optimization, these are declared inline in the
// header.
template <>
inline PrimitiveType NativeToPrimitiveType<bool>() {
  return PrimitiveType::PRED;
}

// Unsigned integer
template <>
inline PrimitiveType NativeToPrimitiveType<uint8>() {
  return PrimitiveType::U8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint16>() {
  return PrimitiveType::U16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint32>() {
  return PrimitiveType::U32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<uint64>() {
  return PrimitiveType::U64;
}

// Signed integer
template <>
inline PrimitiveType NativeToPrimitiveType<int8>() {
  return PrimitiveType::S8;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int16>() {
  return PrimitiveType::S16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int32>() {
  return PrimitiveType::S32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<int64>() {
  return PrimitiveType::S64;
}

// Floating point
template <>
inline PrimitiveType NativeToPrimitiveType<float>() {
  return PrimitiveType::F32;
}

template <>
inline PrimitiveType NativeToPrimitiveType<double>() {
  return PrimitiveType::F64;
}

template <>
inline PrimitiveType NativeToPrimitiveType<half>() {
  return PrimitiveType::F16;
}

template <>
inline PrimitiveType NativeToPrimitiveType<bfloat16>() {
  return PrimitiveType::BF16;
}

// Complex
template <>
inline PrimitiveType NativeToPrimitiveType<complex64>() {
  return PrimitiveType::C64;
}

template <>
inline PrimitiveType NativeToPrimitiveType<complex128>() {
  return PrimitiveType::C128;
}

inline bool IsFloatingPointType(PrimitiveType type) {
  return type == PrimitiveType::F16 || type == PrimitiveType::F32 ||
         type == PrimitiveType::F64 || type == PrimitiveType::BF16;
}

inline bool IsComplexType(PrimitiveType type) {
  return type == PrimitiveType::C64 || type == PrimitiveType::C128;
}

inline bool IsSignedIntegralType(PrimitiveType type) {
  return type == PrimitiveType::S8 || type == PrimitiveType::S16 ||
         type == PrimitiveType::S32 || type == PrimitiveType::S64;
}

inline bool IsUnsignedIntegralType(PrimitiveType type) {
  return type == PrimitiveType::U8 || type == PrimitiveType::U16 ||
         type == PrimitiveType::U32 || type == PrimitiveType::U64;
}

inline bool IsIntegralType(PrimitiveType type) {
  return IsUnsignedIntegralType(type) || IsSignedIntegralType(type);
}

inline int BitWidth(PrimitiveType type) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline PrimitiveType UnsignedIntegralTypeForBitWidth(int64 src_bitwidth) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline PrimitiveType ComplexComponentType(PrimitiveType complex_type) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline const std::string LowercasePrimitiveTypeName(PrimitiveType s) {
  return PrimitiveTypeName(s);
}

}  // namespace primitive_util
}  // namespace lazy_tensors
