#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THCStorage.h"

#include "ATen/ScalarType.h"
#include "ATen/ScalarTypeUtils.h"
#include <atomic>

namespace at {

template <>
struct CTypeToScalarType<__half> : public CTypeToScalarType<Half> {};

}

typedef struct THCStorage
{
    at::ScalarType scalar_type;
    void *data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    THCDeviceAllocator *allocator;
    void *allocatorContext;
    struct THCStorage *view;
    int device;

    template <typename T>
    inline T * data() const {
      auto scalar_type_T = at::CTypeToScalarType<T>::to();
      if (scalar_type != scalar_type_T) {
        AT_ERROR("Attempt to access Storage having data type ", at::toString(scalar_type),
                 " as data type ", at::toString(scalar_type_T));
      }
      return unsafe_data<T>();
    }

    template <typename T>
    inline T * unsafe_data() const {
      return static_cast<T*>(this->data_ptr);
    }
} THCStorage;

THC_API THCStorage* THCStorage_new(THCState *state, at::ScalarType scalar_type);
THC_API THCStorage* THCStorage_newWithSize(THCState *state, at::ScalarType scalar_type, ptrdiff_t size);

THC_API THCStorage* THCStorage_newWithAllocator(THCState *state,
                                        at::ScalarType scalar_type,
                                        ptrdiff_t size,
                                        THCDeviceAllocator* allocator,
                                        void* allocatorContext);

THC_API void THCStorage_retain(THCState *state, THCStorage *storage);

// This exists to have a data-type independent way of freeing (necessary for THPPointer).
THC_API void THCStorage_free(THCState *state, THCStorage *self);

THC_API void THCStorage_resize(THCState *state, THCStorage *storage, ptrdiff_t size);
THC_API int THCStorage_getDevice(THCState* state, const THCStorage* storage);
