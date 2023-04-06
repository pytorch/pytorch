#pragma once

#include <c10/core/impl/cow/shadow_storage.h>
#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>

namespace c10::impl::cow {

// Responsible for managing the copy-on-write simulation state
// machine.
//
// This type manages the following transition:

// 1) A tensor is created. It has a StateMachine instance that is in
//    the StateId::initial state. There are no shadow storages. This tensor and
//    any views created from it will have a null shadow storage.
//
// 2) A lazy copy is created, e.g. from Tensor::reshape(). This
//    transitions from StateId::initializing and then finally to
//    StateId::active. At this point a StateMachine::Impl has been
//    allocated on the heap.
//
//    The Tensor that was the source of the lazy copy, and any views
//    created from it, will still have a null shadow storage pointer,
//    but its actual shadow storage will be in
//    Impl::default_shadow_storage.
class C10_API StateMachine {
 public:
  /** @see cow::ShadowStorage::Generation */
  using Generation = cow::ShadowStorage::Generation;

  // Constructs an instance in the "initial" state.
  StateMachine();

  // Responsible for cleaning up any implementation details.
  ~StateMachine();

  // Gets the current generation of the physical storage.
  //
  // Reminder: no tracking occurs until simulate_lazy_copy() is
  // called.
  auto physical_generation() -> Generation;

  // Bumps the physical generation and updates the appropriate shadow
  // storage to the new value, warning if there is an unexpected gap.
  //
  // This is called by a TensorImpl that is being written to. The
  // shadow storage parameter to this function will be the shadow
  // storage of the calling tensor.
  //
  // The very first tensor, and any views derived from it, have a null
  // shadow storage member as a performance optimization: we try to
  // make this "pay for what you use" as much as possible. A lazy copy
  // may be performed on that tensor *only* after it is created. Thus
  // if we wish to defer allocation of a shadow storage for a tensor,
  // then there will be at least one and possibly many views that will
  // not have a shadow storage.
  //
  // We get around this problem by storing the shadow storage for the
  // initial tensor and its views on the storage object itself
  // (i.e. here). We call this `default_shadow_storage_`.
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
  class Impl;

  /** Gets the underlying Impl, returning null if uninitialized. */
  auto maybe_get_impl() -> Impl*;
  /** Gets the underlying Impl, initializing if uninitialized. */
  auto ensure_initialized() -> Impl&;

  // The current representation of the storage instance.
  std::atomic<Impl*> impl_;
};

} // namespace c10::impl::cow
