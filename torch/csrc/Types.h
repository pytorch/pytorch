#ifndef THP_TYPES_INC
#define THP_TYPES_INC

#include <cstddef>
#include <TH/TH.h>

#ifndef INT64_MAX
#include "stdint.h"
#endif

template <typename T> struct THPTypeInfo {};

namespace torch {

typedef THFloatStorage THVoidStorage;  // all THXXXStorage types are the same.

typedef struct THVoidTensor
{
   int64_t *size;
   int64_t *stride;
   int nDimension;
   THVoidStorage *storage;
   ptrdiff_t storageOffset;
   int refcount;
   char flag;
} THVoidTensor;

}  // namespace torch

#endif
