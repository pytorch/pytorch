#pragma once

#include "ATen/ScalarType.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

struct __half;

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

template <>
struct CTypeToScalarType<__half> {
  static inline at::ScalarType to() { return at::ScalarType::Half; }
};

} // namespace at
