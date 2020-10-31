#pragma once

// STOP!!! Thinking of including this header directly?  Please
// read Note [TH abstraction violation]

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <TH/THStorageFunctions.h>

#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>

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

TH_CPP_API THStorage* THStorage_new();

TH_API void THStorage_retain(THStorage *storage);
TH_API void THStorage_resizeBytes(THStorage* storage, ptrdiff_t size_bytes);
