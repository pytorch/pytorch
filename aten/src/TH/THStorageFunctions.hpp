#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include "THStorageClass.hpp"
#include "THStorageFunctions.h"

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

TH_API THStorage* THStorage_new(at::ScalarType scalar_type);
TH_API THStorage* THStorage_newWithSize(at::ScalarType scalar_type, ptrdiff_t size);
TH_API THStorage* THStorage_newWithAllocator(at::ScalarType scalar_type, ptrdiff_t size,
                                             at::Allocator *allocator);

ptrdiff_t THStorage_size(const THStorage *self);
size_t THStorage_elementSize();
THStorage* THStorage_newWithMapping(at::ScalarType scalar_type, const char *filename, ptrdiff_t size, int flags);
void THStorage_setFlag(THStorage *storage, const char flag);
void THStorage_clearFlag(THStorage *storage, const char flag);
void THStorage_retain(THStorage *storage);
THStorage* THStorage_newWithDataAndAllocator(at::ScalarType scalar_type,
                                             at::DataPtr&& data, ptrdiff_t size,
                                             at::Allocator* allocator);
void THStorage_resize(THStorage *storage, ptrdiff_t size);
void THStorage_swap(THStorage *storage1, THStorage *storage2);

void THStorage_weakRetain(THStorage *weak_storage);
void THStorage_weakFree(THStorage *weak_storage);
THStorage* THStorage_weakLock(THStorage *weak_storage);
