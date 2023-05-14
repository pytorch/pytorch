#pragma once

#include <c10/macros/Macros.h>

namespace c10 {
struct StorageImpl;
}; // namespace c10

namespace c10 {
namespace impl {
namespace cow {

/// Eagerly copies the data marked for lazy-copy in storage.
///
/// Requires: storage has a copy-on-write context.
auto C10_API materialize(StorageImpl& storage) -> void;

} // namespace cow
} // namespace impl
} // namespace c10
