#ifndef THP_TYPES_INC
#define THP_TYPES_INC

#include <cstddef>
#include <TH/TH.h>

#ifndef INT64_MAX
#include <cstdint>
#endif

template <typename T> struct THPTypeInfo {};

namespace torch {

typedef THStorage THVoidStorage;

typedef THTensor THVoidTensor;

}  // namespace torch

#endif
