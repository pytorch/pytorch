#include <c10/core/impl/cow/state_machine.h>

#include <c10/util/Exception.h>

namespace c10::impl {

namespace {

// Applies a function to whichever shadow storage is active.
//
// Note that we templatize on the shadow storage types because they
// may or may not be const. The bare types will always be
// ShadowStorage and ShadowStorageNonIntrusive.
template <typename ShadowStorage, typename DefaultShadowStorage, typename Func>
auto apply_to_active_shadow_storage(
    ShadowStorage* shadow_storage,
    DefaultShadowStorage& default_shadow_storage,
    const Func& func) {
  if (shadow_storage != nullptr) {
    return func(*shadow_storage);
  }
  return func(default_shadow_storage);
}

} // namespace

cow::StateMachine::StateMachine()
    : state_id_(StateId::initial), physical_generation_(0) {}

auto cow::StateMachine::maybe_bump(cow::ShadowStorage* maybe_shadow_storage)
    -> void {
  if (state_id_ != StateId::active) {
    return;
  }
  // else we are in the active state.

  std::lock_guard<std::mutex> lock(mtx_);
  // We check the state again under the mutex because we may have
  // transitioned to StateId::finished in the interim.
  //
  // We load this with the weakest guarantee possible because the
  // mutex acquisition serves as a sufficient memory barrier to ensure
  // we see the most recent value from any previous update which would
  // also be under the lock.
  switch (state_id_.load(std::memory_order_relaxed)) {
    case StateId::initial:
      // We've transitioned back to the null state. This can happen
      // once the last outstanding lazy copy has been safely made. We
      // have nothing to do until a new lazy copy is performed.
      return;

    case StateId::active:
      // Continue on. Nothing can transition out of the active state
      // whilst we hold the lock.
      break;
  }

  // There will always be an active default shadow storage if we are
  // active.
  TORCH_INTERNAL_ASSERT(default_shadow_storage_.has_value());

  // Any created shadow storage should be bound to the physical
  // storage that it was created from. Hence, there should only be a
  // shadow storage on a tensor if its own storage created it. We
  // don't check for the specific matching of the storage and shadow
  // storage, but we do check that the presence relationship holds.
  //
  // TODO: This check should probably be here, but it is actually
  // triggering in the StaticRuntime.autogen_inner test.
  //
  // TORCH_INTERNAL_ASSERT(maybe_shadow_storage != nullptr);

  std::uint64_t physical_generation = ++physical_generation_;
  std::uint64_t shadow_generation = apply_to_active_shadow_storage(
      maybe_shadow_storage, *default_shadow_storage_, [](auto& shadow_storage) {
        return shadow_storage.bump_generation();
      });

  TORCH_INTERNAL_ASSERT(shadow_generation <= physical_generation);
  if (shadow_generation != physical_generation) {
    TORCH_WARN_ONCE(
        "You have written through to both aliases created by calling "
        "reshape(). In the future, reshape() will never create a view but will "
        "instead return a lazily copied tensor. If you wish to preserve the "
        "aliasing properties, you should rewrite your reshape() as a view().");
  }
}

auto cow::StateMachine::physical_generation() -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  return physical_generation_;
}

auto cow::StateMachine::simulate_lazy_copy(
    cow::ShadowStorage* maybe_shadow_storage)
    -> intrusive_ptr<cow::ShadowStorage> {
  // We grab the lock here unconditionally. No need to check the
  // current state first.
  std::lock_guard<std::mutex> lock(mtx_);
  // We load this with the weakest guarantee possible because the
  // mutex acquisition serves as a sufficient memory barrier to ensure
  // we see the most recent value from any previous update which would
  // also be under the lock.
  switch (state_id_.load(std::memory_order_relaxed)) {
    case StateId::initial:
      TORCH_INTERNAL_ASSERT(!default_shadow_storage_.has_value());
      TORCH_INTERNAL_ASSERT(physical_generation_ == 0);
      TORCH_INTERNAL_ASSERT(
          state_id_.exchange(StateId::active, std::memory_order_relaxed) ==
          StateId::initial);
      default_shadow_storage_.emplace(0);
      return make_intrusive<cow::ShadowStorage>(0);

    case StateId::active:
      TORCH_INTERNAL_ASSERT(default_shadow_storage_.has_value());
      return make_intrusive<cow::ShadowStorage>(apply_to_active_shadow_storage(
          maybe_shadow_storage,
          *default_shadow_storage_,
          [](auto const& shadow_storage) {
            return shadow_storage.generation();
          }));
  }
  TORCH_INTERNAL_ASSERT(false); // unreachable
}

} // namespace c10::impl
