#pragma once

#include <c10/core/ScalarType.h>

namespace c10 {

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

} // namespace c10
