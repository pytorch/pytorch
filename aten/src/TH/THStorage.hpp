#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THStorage.h"

#include <ATen/ScalarType.h>
#include <ATen/ScalarTypeUtils.h>
#include "THTypeConversion.hpp"
#include <atomic>

// Note [Weak references for intrusive refcounting]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Here's the scheme:
//
//  - refcount == number of strong references to the object
//    weakcount == number of weak references to the object,
//      plus one more if refcount > 0
//
//  - THStorage stays live as long as there are any strong
//    or weak pointers to it (weakcount > 0, since strong
//    references count as a +1 to weakcount)
//
//  - finalizers are called and data_ptr is deallocated when refcount == 0
//
//  - Once refcount == 0, it can never again be > 0 (the transition
//    from > 0 to == 0 is monotonic)
//
//  - When you access THStorage via a weak pointer, you must
//    atomically increment the use count, if it is greater than 0.
//    If it is not, you must report that the storage is dead.
//

struct THFinalizer {
  virtual void operator()() = 0;
  virtual ~THFinalizer() {};
};

typedef struct THStorage
{
    at::ScalarType scalar_type;
    at::DataPtr data_ptr;
    ptrdiff_t size;
    std::atomic<int> refcount;
    std::atomic<int> weakcount;
    char flag;
    at::Allocator *allocator;
    std::unique_ptr<THFinalizer> finalizer;

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
      return static_cast<T*>(this->data_ptr.get());
    }
} THStorage;

TH_API THStorage* THStorage_new(at::ScalarType scalar_type);
TH_API THStorage* THStorage_newWithSize(at::ScalarType scalar_type, ptrdiff_t size);
TH_API THStorage* THStorage_newWithAllocator(at::ScalarType scalar_type, ptrdiff_t size,
                                             at::Allocator *allocator);

TH_API ptrdiff_t THStorage_size(const THStorage *self);
TH_API size_t THStorage_elementSize();
TH_API THStorage* THStorage_newWithMapping(at::ScalarType scalar_type, const char *filename, ptrdiff_t size, int flags);
TH_API void THStorage_setFlag(THStorage *storage, const char flag);
TH_API void THStorage_clearFlag(THStorage *storage, const char flag);
TH_API void THStorage_retain(THStorage *storage);
TH_API THStorage* THStorage_newWithDataAndAllocator(at::ScalarType scalar_type,
                                                    at::DataPtr&& data, ptrdiff_t size,
                                                    at::Allocator* allocator);
TH_API void THStorage_resize(THStorage *storage, ptrdiff_t size);
TH_API void THStorage_swap(THStorage *storage1, THStorage *storage2);

TH_API void THStorage_weakRetain(THStorage *weak_storage);
TH_API void THStorage_weakFree(THStorage *weak_storage);
TH_API THStorage* THStorage_weakLock(THStorage *weak_storage);
