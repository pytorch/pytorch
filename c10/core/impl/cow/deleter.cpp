#include <c10/core/impl/cow/deleter.h>

#include <c10/core/impl/cow/context.h>

namespace c10::impl {

/// Deletes a copy-on-write context.
///
/// Requires: ctx is cow::Context.
auto cow::delete_context(void* ctx) -> void {
  static_cast<cow::Context*>(ctx)->decrement_refcount();
}

} // namespace c10::impl
