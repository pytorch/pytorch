#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct StorageImpl;
class DataPtr;
}; // namespace c10

namespace c10::impl::cow {

// Creates a Copy-on-write (COW) clone of the given storage. This will also
// convert the given storage into a COW storage if it is not COW already.
//
// Converting the storage into a COW storage will not be successful if the
// storage's DataPtr has some context (`DataPtr::get_context()`) which is not
// equal to the data pointer (`DataPtr::get()`). In this case, a nullptr is
// returned.
C10_API c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage);

// Creates a view of the given storage. However, COW behavior is simulated,
// and if any writes/reads cause user-observable behavior to diverge from
// that of COW storages, then a warning is raised.
C10_API c10::intrusive_ptr<StorageImpl> simulate_lazy_clone_storage(
    StorageImpl& storage);

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
C10_API void set_future_copy_instead_of_conditional_view(bool mode);
C10_API bool get_future_copy_instead_of_conditional_view();

} // namespace c10::impl::cow
