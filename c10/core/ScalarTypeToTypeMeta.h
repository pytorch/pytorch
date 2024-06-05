#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <c10/util/typeid.h>

// these just expose TypeMeta/ScalarType bridge functions in c10
// TODO move to typeid.h (or codemod away) when TypeMeta et al
// are moved from caffe2 to c10 (see note at top of typeid.h)

namespace c10 {

/**
 * convert ScalarType enum values to TypeMeta handles
 */
inline caffe2::TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
  return caffe2::TypeMeta::fromScalarType(scalar_type);
}

/**
 * convert TypeMeta handles to ScalarType enum values
 */
inline ScalarType typeMetaToScalarType(caffe2::TypeMeta dtype) {
  return dtype.toScalarType();
}

/**
 * typeMetaToScalarType(), lifted to optional
 */
inline optional<at::ScalarType> optTypeMetaToScalarType(
    optional<caffe2::TypeMeta> type_meta) {
  if (!type_meta.has_value()) {
    return c10::nullopt;
  }
  return type_meta->toScalarType();
}

/**
 * convenience: equality across TypeMeta/ScalarType conversion
 */
inline bool operator==(ScalarType t, caffe2::TypeMeta m) {
  return m.isScalarType(t);
}

inline bool operator==(caffe2::TypeMeta m, ScalarType t) {
  return t == m;
}

inline bool operator!=(ScalarType t, caffe2::TypeMeta m) {
  return !(t == m);
}

inline bool operator!=(caffe2::TypeMeta m, ScalarType t) {
  return !(t == m);
}

} // namespace c10
