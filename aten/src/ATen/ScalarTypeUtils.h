#pragma once

#include "ATen/ScalarType.h"

namespace at {

template <typename T>
struct CTypeToScalarType {
};

#define DEFINE_TO_SCALAR_TYPE(ct, st, _2)                          \
template <>                                                        \
struct CTypeToScalarType<ct> {                                     \
  static inline at::ScalarType to() { return at::ScalarType::st; } \
};
AT_FORALL_SCALAR_TYPES(DEFINE_TO_SCALAR_TYPE)
#undef DEFINE_TO_SCALAR_TYPE

} // namespace at
