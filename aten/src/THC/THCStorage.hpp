#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THCStorage.h"
#include <TH/THStorage.hpp>

#include "ATen/ScalarType.h"
#include "ATen/ScalarTypeUtils.h"
#include <atomic>

namespace at {

template <>
struct CTypeToScalarType<__half> : public CTypeToScalarType<Half> {};

}

THC_API THCStorage* THCStorage_new(THCState *state, at::ScalarType scalar_type);
THC_API THCStorage* THCStorage_newWithSize(THCState *state, at::ScalarType scalar_type, ptrdiff_t size);

THC_API THCStorage* THCStorage_newWithAllocator(THCState *state,
                                        at::ScalarType scalar_type,
                                        ptrdiff_t size,
                                        at::Allocator* allocator);

THC_API void THCStorage_retain(THCState *state, THCStorage *storage);

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
THC_API void THCStorage_free(THCState *state, THCStorage *self);

THC_API void THCStorage_resize(THCState *state, THCStorage *storage, ptrdiff_t size);
THC_API int THCStorage_getDevice(THCState* state, const THCStorage* storage);

THC_API THCStorage* THCStorage_newWithDataAndAllocator(
  THCState *state, at::ScalarType scalar_type,
  at::DataPtr&& data, ptrdiff_t size,
  at::Allocator* allocator);
