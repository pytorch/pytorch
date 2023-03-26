#pragma once

#include <c10/core/impl/cow/shadow_storage.h>
#include <c10/macros/Macros.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <mutex>

namespace c10::impl::cow {

// Responsible for managing the copy-on-write simulation state
// machine.
class C10_API StateMachine {
 public:
  // Constructs an instance in the "initial" state.
  StateMachine();

  // Gets the current generation of the physical storage.
  auto physical_generation() -> std::uint64_t;

  // Bumps the generation if the shadow storage is non-null. If
  // non-null, shadow storage must be the result of a previous call to
  // simulate_copy_on_write on this instance.
  auto maybe_bump(cow::ShadowStorage* maybe_shadow_storage) -> void;

  // Simulates a lazy copy a tensor that owns the shadow storage.
  //
  // maybe_shadow_storage comes from the tensor that is being lazily
  // copied. This may be null if this is the first lazy copy taking
  // place, or if the lazy copy is being performed on a tensor that
  // was part of the original tensors that share a view.
  //
  // The generation of the output will come from:
  // 1) maybe_shadow_storage->generation(), if non-null.
  // 2) this->default_shadow_storage_->generation(), if it is set.
  // 3) physical_generation_, i.e. 0, if this is the first lazy copy.
  auto simulate_lazy_copy(cow::ShadowStorage* maybe_shadow_storage)
      -> intrusive_ptr<cow::ShadowStorage>;

 private:
  // Identifies where a storage instance is in the lifecycle of
  // tracking simulated lazy copies.
  enum class StateId {
    // This is the initial state. It may not be returned to and
    // indicates that the storage has never had a lazy copy requested
    // of it.
    //
    // INVARIANT: not default_shadow_storage_.has_value()
    initial,

    // This is transitioned to when the first copy on write takes
    // place.
    //
    // INVARIANT: default_shadow_storage_.has_value()
    active,

    // TODO Consider adding an "inactive" state. An inactive state
    //      would occur when the last outstanding lazy copy goes away
    //      or has been materialized. At that point, we may stop
    //      tracking generations.
    //
    //      We may want to do this because we've already warned or
    //      because we only had a temporary lazy copy that shouldn't
    //      affect the performance of the program for the remainder of
    //      its lifetime.
    //
    //      DANGER! There could be outstanding shadow storages. For
    //      example, if the last set of tensors sharing a view
    //      remaining was not using the default shadow storage, it
    //      will continue holding the instances that they share.
    //
    //      There are a few potential solutions to this. One is for
    //      the state machine to be the exclusive owner of the shadow
    //      storages and for tensors to only hold weak pointers to
    //      them.
    //
    //      Another is for tensors to hold a storage id and an index
    //      and those can be invalidated by increasing the storage id
    //      whenever we transition to StateId::inactive. A nice
    //      property of this is that it would generalize the design
    //      decision around the "default shadow storage".
  };

  // The current state of the storage instance.
  std::atomic<StateId> state_id_;

  // Guards all the state.
  std::mutex mtx_;
  // How many writes have been applied to the storage.
  std::uint64_t physical_generation_;
  // The shadow storage to use for any tensors that don't have
  // one. This situation is common, and will be true for tensors and
  // views thereof created before any copy on writes.
  //
  // Note: this would like to be std::optional, but it can't be
  // because of torchdeploy incompatibility.
  //
  // See https://github.com/pytorch/multipy/issues/312
  c10::optional<cow::ShadowStorageNonIntrusive> default_shadow_storage_;
};

} // namespace c10::impl::cow
