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
  using Generation = std::int64_t;

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

} // namespace c10::impl::cow
