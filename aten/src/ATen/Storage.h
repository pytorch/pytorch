#pragma once

#include <ATen/StorageImpl.h>

namespace at {

// OPtions: Use retainable
// Use strorage ptr == unique ptr <custom destructor calls release on storage>
//  - custom destructor will just decrease refcount
// Could rename this to StorageImplImpl and then create StorageImpl that maintains StorageImplImpl and decrease refcount


// StorageImpl/StorageImplImpl thing
// Inherit from retainable
// StorageImpl has deleter, ; assignment and move deleted, pretty much like StorageImpl and THStorageImpl before
// - - don't have to deleted unique ptr - will use intrusive and retainable


// Details:
// Forward the constructor just like before
// Deconstructor just calls free on StorageImpl
// Create a StorageImpl free function

struct Storage {
  Storage() : storage_impl_(nullptr){};
  Storage(StorageImpl* storage_impl) : storage_impl_(storage_impl){};
  ~Storage();
  StorageImpl* storage_impl_;
  void* unsafeGetTH(bool retain_) const {
    if (retain_)
      storage_impl_->retain();
    return storage_impl_;
  }
};

} // namespace at
