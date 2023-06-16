// This is its own header to minimize code visible in other public
// headers in the system. This is beneficial for compilation times as
// well as to avoid issues with internal Meta builds that aren't using
// C++17.

#pragma once

#include <c10/macros/Export.h>

namespace c10 {
namespace impl {
namespace cow {

/// Deletes a copy-on-write context.
///
/// Requires: ctx is cow::Context.
auto C10_API delete_context(void* ctx) -> void;

} // namespace cow
} // namespace impl
} // namespace c10
