#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>

namespace c10 {

/**
 * convert ScalarType enum values to TypeMeta handles
 */
static inline caffe2::TypeMeta scalarTypeToTypeMeta(ScalarType scalar_type) {
#define DEFINE_CASE(ctype, name) \
  case ScalarType::name:         \
    return caffe2::TypeMeta::Make<ctype>();

  switch (scalar_type) {
    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CASE)
    case ScalarType::Undefined:
      return caffe2::TypeMeta();
    default:
      AT_ERROR("Unrecognized Scalartype ", scalar_type, " (please report this error)");
  }
#undef DEFINE_CASE
}

/**
 * convert TypeMeta handles to ScalarType enum values
 * Caution: extremely hot
 */
static inline ScalarType typeMetaToScalarType(caffe2::TypeMeta dtype) {
  if (auto scalar_type = dtype.scalarTypeOpt()) {
    return *scalar_type;
  }
  AT_ERROR(
      "Unsupported TypeMeta in ATen: ", dtype, " (please report this error)");
}

/**
 * typeMetaToScalarType(), lifted to optional
 */
inline optional<at::ScalarType> optTypeMetaToScalarType(optional<caffe2::TypeMeta> type_meta) {
  if (!type_meta.has_value()) {
    return c10::nullopt;
  }
  return typeMetaToScalarType(*type_meta);
}

/**
 * convenience: equality across TypeMeta/ScalarType conversion
 */
static inline bool operator==(ScalarType t, caffe2::TypeMeta m) {
  if (auto mt = m.scalarTypeOpt()) {
    return (*mt) == t;
  }
  return false;
}

static inline bool operator==(caffe2::TypeMeta m, ScalarType t) {
  return t == m;
}

} // namespace c10
