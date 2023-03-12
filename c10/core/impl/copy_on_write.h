#pragma once

#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <optional>

namespace c10 {
struct Storage;
struct StorageImpl;
}; // namespace c10

namespace c10::impl {

// Ensures storage is copy-on-write, returning a new impl sharing the
// data.
//
// This will try to convert the storage to use a CopyOnWriteContext if
// it is not already. Returns null only if Storage does not have a
// CopyOnWriteContext upon completion.
auto C10_API make_copy_on_write(Storage const& storage)
    -> c10::intrusive_ptr<StorageImpl>;

// Gets the refcount if the storage is copy-on-write.
auto TORCH_API copy_on_write_refcount(Storage const& storage)
    -> std::optional<std::int64_t>;

} // namespace c10::impl
