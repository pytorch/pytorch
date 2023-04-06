#pragma once

// This file provides two public types: ShadowStorage and
// ShadowStorageNonIntrusive.
//
// See ShadowStorageImpl to understand the interface of these types.

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

#include <atomic>
#include <cstdint>
#include <type_traits>

namespace c10::impl::cow {

// Represents the shadow storage for a tensor.
//
// For simulated lazy copies, the shadow storage represents the what
// the storage would have been for a tensor that was lazily
// copied. However, when simulating this, we can't actually replace
// the storage because we would violate a core implementation
// invariant that tensors which are views of each other share the same
// storage.
//
// However, the fidelity of this check is limited by the extent to
// which we have instrumented operations as reading or writing to
// storage.
//
// We use a monotonically increasing generation number to track
// modifications to storage.
//
// Note that we templatize on whether or not we are eligible to be
// allocated into an intrusive_ptr. It *seems* as though ASan is
// unhappy if we stack-allocate an object that inherits from
// c10::intrusive_ptr_target.
class C10_API ShadowStorage final : private intrusive_ptr_target {
 public:
  /** The type of the generation number. */
  using Generation = std::int32_t;

  // Creates an instance from an existing storage generation.
  explicit ShadowStorage(Generation generation) noexcept;

  // Gets the current generation.
  auto generation() const noexcept -> Generation;

  // Sets the generation from the physical generation, warning if they
  // do not agree.
  auto update_from_physical_generation(Generation physical_generation) noexcept
      -> void;

 private:
  // From the user's perspective, copies are fully distinct and
  // require no external synchronization, so we need to ensure this
  // field's concurrency is properly managed.
  std::atomic<Generation> generation_;

  friend intrusive_ptr<ShadowStorage>;
};

// A mixin to be used on TensorImpl that gives it a shadow storage.
//
// We add this as a mixin because we desire the ability to toggle
// shadow storage tracking at compilation time. If disabled, this
// becomes an empty class and the empty base class optimization will
// make this add no weight to TensorImpl.
class C10_API ShadowStorageMixin {
 protected:
  // Initialize the field from a possibly null shadow_storage.
  explicit ShadowStorageMixin(intrusive_ptr<cow::ShadowStorage> shadow_storage);

  // Gets the possibly null shadow storage.
  //
  // Use this by default.
  auto shadow_storage() const -> cow::ShadowStorage*;

  // Gets a reference to the possibly null shadow storage.
  //
  // Only use this if you wish to have another reference to the same
  // instance, for example, when taking a view.
  auto shadow_storage_ref() const -> intrusive_ptr<cow::ShadowStorage>;

 private:
#if defined(PYTORCH_INSTRUMENT_COW_TENSOR)
  // Invariant: this is always null if a copy on write was never
  // requested.
  //
  // This *may* be null if a copy on write was requested and this
  // tensor is part of the original view family. Subsequent view
  // families will have this set, but the original one only gets its
  // value from the storage.
  //
  // This is asymmetrical, but it allows us to avoid the allocation
  // and any refcount bumps until we actually need them.
  intrusive_ptr<impl::cow::ShadowStorage> shadow_storage_;
#endif
};

} // namespace c10::impl::cow
