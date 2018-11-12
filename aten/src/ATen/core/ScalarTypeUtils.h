#pragma once

#include "ATen/core/ScalarType.h"

namespace at {

template <typename T>
struct CTypeToScalarType {
};

#define DEFINE_TO_SCALAR_TYPE(ct, st, _2)                          \
template <>                                                        \
struct CTypeToScalarType<ct> {                                     \
  static inline at::ScalarType to() { return at::ScalarType::st; } \
};
AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_TO_SCALAR_TYPE)
#undef DEFINE_TO_SCALAR_TYPE

} // namespace at
