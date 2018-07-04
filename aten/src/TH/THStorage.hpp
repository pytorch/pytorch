#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THStorage.h"

#include <ATen/ScalarType.h>
#include <ATen/ScalarTypeUtils.h>
#include "THTypeConversion.hpp"
#include <atomic>

typedef struct THStorage
{
    at::Backend backend; // kCPU or kCUDA only
    at::ScalarType scalar_type;
    void *data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    char flag;
    void *allocatorVoidPtr; // Either THDeviceAllocator or THCDeviceAllocator
    void *allocatorContext;
    struct THStorage *view;
    int device;

    template <typename T>
    inline T * data() const {
      auto scalar_type_T = at::CTypeToScalarType<th::from_type<T>>::to();
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
} THStorage;

TH_API THStorage* THStorage_new(at::ScalarType scalar_type);
TH_API THStorage* THStorage_newWithSize(at::ScalarType scalar_type, ptrdiff_t size);
TH_API THStorage* THStorage_newWithAllocator(at::ScalarType scalar_type, ptrdiff_t size,
                                             THAllocator *allocator,
                                             void *allocatorContext);

ptrdiff_t THStorage_size(const THStorage *self);
size_t THStorage_elementSize();
THStorage* THStorage_newWithMapping(at::ScalarType scalar_type, const char *filename, ptrdiff_t size, int flags);
void THStorage_setFlag(THStorage *storage, const char flag);
void THStorage_clearFlag(THStorage *storage, const char flag);
void THStorage_retain(THStorage *storage);
int THStorage_retainIfLive(THStorage *storage);
THStorage* THStorage_newWithData(at::ScalarType scalar_type, void *data, ptrdiff_t size);
THStorage* THStorage_newWithDataAndAllocator(at::ScalarType scalar_type,
                                             void* data, ptrdiff_t size,
                                             THAllocator* allocator,
                                             void* allocatorContext);
void THStorage_resize(THStorage *storage, ptrdiff_t size);
void THStorage_swap(THStorage *storage1, THStorage *storage2);
