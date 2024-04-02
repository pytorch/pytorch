#pragma once

#include <c10/core/Storage.h>
#include <c10/util/UniqueVoidPtr.h>

#include <atomic>
#include <memory>

namespace c10 {

// A RefcountedDeleterContext object is used as the `ctx` argument for DataPtr
// to implement a shared DataPtr. Normally, a DataPtr is unique, but we use
// this custom context and the `refcounted_deleter` function below to make the
// DataPtr act like a non-unique DataPtr. This context object holds onto an
// inner context and deleter function which handle the actual deletion of the
// data when the refcount reaches 0.
//
// This shared DataPtr feature is only used when storages are shared between
// multiple Python interpreters in MultiPy. Before storages had PyObject
// preservation, interpreters could just share the same StorageImpl instance.
// But now a StorageImpl can only be associated with one interpreter in order
// to properly manage a zombie PyObject. So we share storages across Python
// interpreters by creating a different StorageImpl instance for each one, but
// they all point to the same data.
struct C10_API RefcountedDeleterContext {
  RefcountedDeleterContext(void* other_ctx, c10::DeleterFnPtr other_deleter)
      : other_ctx(other_ctx, other_deleter), refcount(1) {}

  std::unique_ptr<void, c10::DeleterFnPtr> other_ctx;
  std::atomic_int refcount;
};

// `refcounted_deleter` is used as the `ctx_deleter` for DataPtr to implement
// a shared DataPtr.
//
// Warning: This should only be called on a pointer to
// a RefcountedDeleterContext that was allocated on the heap with `new`,
// because when the refcount reaches 0, the context is deleted with `delete`
C10_API void refcounted_deleter(void* ctx_);

// If the storage's DataPtr does not use `refcounted_deleter`, replace it with
// a DataPtr that does, so it can be shared between multiple StorageImpls
C10_API void maybeApplyRefcountedDeleter(const c10::Storage& storage);

// Create a new StorageImpl that points to the same data. If the original
// StorageImpl's DataPtr does not use `refcounted_deleter`, it will be replaced
// with one that does
C10_API c10::Storage newStorageImplFromRefcountedDataPtr(
    const c10::Storage& storage);

} // namespace c10
