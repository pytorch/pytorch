#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct StorageImpl;
class DataPtr;
} // namespace c10

namespace c10::impl::cow {

// If `get_future_lazy_clone() == true`, creates a Copy-on-write (COW) clone of
// the given storage. This will also convert the given storage into a COW
// storage if it is not COW already.
//
// If `get_future_lazy_clone() == false`, creates a simulated COW (COWSim) view
// of the given storage. This will also convert the given storage into a COWSim
// storage if it is not already. If any writes/reads cause user-observable
// behavior to diverge from that of COW storages, then a warning is raised.
//
// Converting the storage into a COW or COWSim storage will not be successful if
// the storage's DataPtr has some context (`DataPtr::get_context()`) which is
// not equal to the data pointer (`DataPtr::get()`). In this case, a nullptr is
// returned.
C10_API c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage,
    bool future);

// Check if a storage has a simple DataPtr with no abnormal context
C10_API bool has_simple_data_ptr(const c10::StorageImpl& storage);

// Check if a DataPtr is COW
C10_API bool is_cow_data_ptr(const c10::DataPtr& data_ptr);

// Check if a DataPtr is simulated COW
C10_API bool is_cowsim_data_ptr(const c10::DataPtr& data_ptr);

// Eagerly copies a COW storage's data, turning it into a non-COW storage.
C10_API void materialize_cow_storage(StorageImpl& storage);

C10_API void check_cowsim_write(StorageImpl& storage);

C10_API void check_cowsim_read(const StorageImpl& storage);

// Enables future behavior to make operators which currently conditionally
// return either a copy or a view always return a copy instead.
C10_API void set_future_lazy_clone(bool mode);
C10_API bool get_future_lazy_clone();

class C10_API FutureLazyCloneGuard {
 public:
  FutureLazyCloneGuard(bool mode) : mode_restore(cow::get_future_lazy_clone()) {
    cow::set_future_lazy_clone(mode);
  }
  ~FutureLazyCloneGuard() {
    cow::set_future_lazy_clone(mode_restore);
  }
  FutureLazyCloneGuard(FutureLazyCloneGuard const&) = delete;
  FutureLazyCloneGuard& operator=(FutureLazyCloneGuard const&) = delete;
  FutureLazyCloneGuard(FutureLazyCloneGuard&&) = delete;
  FutureLazyCloneGuard& operator=(FutureLazyCloneGuard&&) = delete;

 private:
  bool mode_restore;
};

} // namespace c10::impl::cow
