#ifndef THP_TYPES_INC
#define THP_TYPES_INC

#include <cstddef>

#ifndef INT64_MAX
#include "stdint.h"
#endif

template <typename T> struct THPTypeInfo {};

namespace torch {

typedef struct THVoidStorage
{
  void *data;
  ptrdiff_t size;
  int refcount;
  char flag;
  void *allocator;
  void *allocatorContext;
  THVoidStorage *view;
} THVoidStorage;

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
