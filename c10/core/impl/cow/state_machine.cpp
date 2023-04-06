#include <c10/core/impl/cow/state_machine.h>

#include <c10/util/Exception.h>

namespace c10::impl {

namespace {

constexpr cow::StateMachine::Generation uninitialized_sentinel = -1;

} // namespace

cow::StateMachine::StateMachine()
    : physical_generation_(uninitialized_sentinel),
      default_shadow_storage_(0) {}

auto cow::StateMachine::physical_generation() -> c10::optional<Generation> {
  Generation physical_generation =
      physical_generation_.load(std::memory_order_relaxed);
  if (physical_generation == uninitialized_sentinel) {
    return c10::nullopt;
  }
  TORCH_INTERNAL_ASSERT(physical_generation >= 0);
  return physical_generation;
}

namespace {

// Applies a function to whichever shadow storage is active.
//
// Note that we templatize on the shadow storage types because they
// may or may not be const. The bare types will always be
// ShadowStorage and ShadowStorageNonIntrusive.
auto active_shadow_storage(
    cow::ShadowStorage* shadow_storage,
    cow::ShadowStorage& default_shadow_storage) -> cow::ShadowStorage& {
  if (shadow_storage != nullptr) {
    return *shadow_storage;
  }
  return default_shadow_storage;
}

} // namespace

auto cow::StateMachine::maybe_bump(cow::ShadowStorage* maybe_shadow_storage)
    -> void {
  // Use Release-Consume ordering to ensure that any initialization
  // will sequence before any read of that impl.
  Generation physical_generation =
      physical_generation_.load(std::memory_order_consume);
  if (physical_generation == uninitialized_sentinel) {
    // Any created shadow storage should be bound to the physical
    // storage that it was created from. Hence, there should only be a
    // shadow storage on a tensor if its own storage created it. We
    // don't check for the specific matching of the storage and shadow
    // storage, but we do check that the presence relationship holds.
    //
    TORCH_INTERNAL_ASSERT(maybe_shadow_storage == nullptr);
    return;
  }

  TORCH_INTERNAL_ASSERT(
      physical_generation != std::numeric_limits<Generation>::max());

  physical_generation = ++physical_generation_;
  active_shadow_storage(maybe_shadow_storage, default_shadow_storage_)
      .update_from_physical_generation(physical_generation);
}

auto cow::StateMachine::simulate_lazy_copy(
    cow::ShadowStorage* maybe_shadow_storage)
    -> intrusive_ptr<cow::ShadowStorage> {
  // We can get away with a relaxed read here because if we believe we
  // are uninitialized, we will just attempt to transition to the
  // initialized state, and in the worst case, we just re-read the
  // generation more strongly.
  Generation physical_generation =
      physical_generation_.load(std::memory_order_relaxed);
  if (physical_generation == uninitialized_sentinel) {
    if (physical_generation_.compare_exchange_strong(physical_generation, 0)) {
      // We successfully initialized, so we know our
      // physical_generation is 0.
      physical_generation = 0;
    } else {
      // Another thread initialized, but the current physical
      // generation is now in physical_generation.
      TORCH_INTERNAL_ASSERT(physical_generation != uninitialized_sentinel);
    }
  }
  TORCH_INTERNAL_ASSERT(physical_generation != uninitialized_sentinel);

  Generation shadow_generation =
      active_shadow_storage(maybe_shadow_storage, default_shadow_storage_)
          .generation();
  TORCH_INTERNAL_ASSERT(shadow_generation <= physical_generation);
  return make_intrusive<cow::ShadowStorage>(shadow_generation);
}

} // namespace c10::impl
