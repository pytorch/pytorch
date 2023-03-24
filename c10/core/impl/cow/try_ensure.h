#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct Storage;
struct StorageImpl;
}; // namespace c10

namespace c10::impl::cow {

// Ensures storage is copy-on-write, returning a new impl sharing the
// data.
//
// This will try to convert the storage to use a CopyOnWriteContext if
// it is not already. Returns null only if Storage does not have a
// CopyOnWriteContext upon completion.
auto C10_API try_ensure(Storage const& storage) -> intrusive_ptr<StorageImpl>;

} // namespace c10::impl::cow
