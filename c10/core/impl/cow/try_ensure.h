#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct StorageImpl;
}; // namespace c10

namespace c10::impl::cow {

// Ensures storage is copy-on-write, returning a new StorageImpl
// sharing the data.
//
// The result is suitable for creating a new Tensor that is logically
// distinct but shares data still.
//
// This will try to convert the storage to use a copy-on-write context
// if it is not already. Returns null only if Storage does not have a
// copy-on-write context upon completion.
auto C10_API try_ensure(StorageImpl& storage) -> intrusive_ptr<StorageImpl>;

} // namespace c10::impl::cow
