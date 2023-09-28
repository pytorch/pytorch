#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct StorageImpl;
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

} // namespace c10::impl::cow
