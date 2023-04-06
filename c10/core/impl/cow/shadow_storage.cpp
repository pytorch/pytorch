#include <c10/core/impl/cow/shadow_storage.h>

#include <c10/util/Exception.h>

#include <limits>

namespace c10::impl {

cow::ShadowStorage::ShadowStorage(Generation generation) noexcept
    : generation_(generation) {}

auto cow::ShadowStorage::generation() const noexcept -> Generation {
  return generation_;
}

namespace {

// Write the warning from a function because if we write it from a
// template, it will warn once for every template instantiation.
auto warn_on_write_after_write() -> void {
  TORCH_WARN_ONCE(
      "You have written through to both aliases created by calling reshape(). "
      "In the future, reshape() will never create a view but will return a "
      "lazily copied tensor. If you wish to preserve the aliasing properties, "
      "you should rewrite your reshape() as a view().");
}

} // namespace

auto cow::ShadowStorage::update_from_physical_generation(
    Generation physical_generation) noexcept -> void {
  TORCH_INTERNAL_ASSERT(physical_generation != 0);

  Generation shadow_generation = generation_.exchange(physical_generation);
  TORCH_INTERNAL_ASSERT(shadow_generation <= physical_generation);

  if (shadow_generation + 1 != physical_generation) {
    warn_on_write_after_write();
  }
}

} // namespace c10::impl
