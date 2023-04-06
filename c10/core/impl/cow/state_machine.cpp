#include <c10/core/impl/cow/state_machine.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>

#include <mutex>

namespace c10::impl {

class cow::StateMachine::Impl {
 public:
  // Bumps the physical and shadow generation numbers.
  auto bump(cow::ShadowStorage* maybe_shadow_storage) -> void;

  /** @see cow::StateMachine::simulate_lazy_copy */
  auto simulate_lazy_copy(cow::ShadowStorage* maybe_shadow_storage)
      -> intrusive_ptr<cow::ShadowStorage>;

 private:
  friend class cow::StateMachine;

  // Guards all the state.
  std::mutex mtx_;
  // How many writes have been applied to the storage.
  //
  // How are physical and shadow generation numbes distinct?
  //
  // * the physical generation is how many writes have occured to a storage
  // * the shadow generation is how many writes have occurred to a set
  //   of tensors which share a view
  //
  // Since storage is 1:many with sets of tensors sharing a view, we
  // expect physical_generation to always be greater than or equal to
  // the shadow generation number.
  cow::StateMachine::Generation physical_generation_ = 0;
  // The shadow storage to use for any tensors that don't have
  // one. This situation is common, and will be true for tensors and
  // views thereof created before any copy on writes.
  cow::ShadowStorage default_shadow_storage_{physical_generation_};
};

cow::StateMachine::StateMachine() : impl_(nullptr) {}

cow::StateMachine::~StateMachine() {
  // If we created an impl, we are responsible for cleaning this up
  // here.
  delete impl_.load();
}

auto cow::StateMachine::physical_generation() -> Generation {
  Impl* impl = maybe_get_impl();
  return impl != nullptr ? impl->physical_generation_ : 0;
}

auto cow::StateMachine::maybe_bump(cow::ShadowStorage* maybe_shadow_storage)
    -> void {
  Impl* impl = maybe_get_impl();
  if (impl == nullptr) {
    // Any created shadow storage should be bound to the physical
    // storage that it was created from. Hence, there should only be a
    // shadow storage on a tensor if its own storage created it. We
    // don't check for the specific matching of the storage and shadow
    // storage, but we do check that the presence relationship holds.
    //
    TORCH_INTERNAL_ASSERT(maybe_shadow_storage == nullptr);
    return;
  }

  impl->bump(maybe_shadow_storage);
}

auto cow::StateMachine::simulate_lazy_copy(
    cow::ShadowStorage* maybe_shadow_storage)
    -> intrusive_ptr<cow::ShadowStorage> {
  return ensure_initialized().simulate_lazy_copy(maybe_shadow_storage);
}

auto cow::StateMachine::maybe_get_impl() -> cow::StateMachine::Impl* {
  // Use Release-Consume ordering to ensure that any creation of an
  // impl will sequence before any read of that impl.
  return impl_.load(std::memory_order_consume);
}

auto cow::StateMachine::ensure_initialized() -> cow::StateMachine::Impl& {
  // We can get away with a relaxed load here because in the worst
  // case we'll get the real value below instead.
  if (Impl* impl = impl_.load(std::memory_order_relaxed); impl != nullptr) {
    return *impl;
  }

  // We are not initialized. Speculatively create an implementation
  // and see if we win.
  auto new_impl = std::make_unique<cow::StateMachine::Impl>();

  Impl* actual_impl = nullptr;
  // Use Release-Acquire ordering to ensure that any creation of an
  // impl will sequence before any read of that impl.
  if (!impl_.compare_exchange_strong(
          actual_impl, &*new_impl, std::memory_order_acq_rel)) {
    // We raced with another initializer and lost. Return the
    // current impl, which is asserted to be active inside
    // as_state_machine_impl().
    TORCH_INTERNAL_ASSERT(actual_impl != nullptr);
    return *actual_impl;
  }

  return *new_impl.release(); // owned by impl_ now
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

auto cow::StateMachine::Impl::bump(cow::ShadowStorage* maybe_shadow_storage)
    -> void {
  std::lock_guard<std::mutex> lock(mtx_);
  TORCH_INTERNAL_ASSERT(
      physical_generation_ != std::numeric_limits<Generation>::max());
  Generation physical_generation = ++physical_generation_;
  active_shadow_storage(maybe_shadow_storage, default_shadow_storage_)
      .update_from_physical_generation(physical_generation);
}

auto cow::StateMachine::Impl::simulate_lazy_copy(
    cow::ShadowStorage* maybe_shadow_storage)
    -> intrusive_ptr<cow::ShadowStorage> {
  // We grab the lock here unconditionally. No need to check the
  // current state first.
  std::lock_guard<std::mutex> lock(mtx_);
  return make_intrusive<cow::ShadowStorage>(
      active_shadow_storage(maybe_shadow_storage, default_shadow_storage_)
          .generation());
}

} // namespace c10::impl
