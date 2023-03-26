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

namespace c10::impl::cow::detail {

// An empty struct to use as the base class for a non-intrusive
// ShadowStorageImpl.
struct Empty {};

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
template <bool intrusive = true>
class C10_API ShadowStorageImpl final
    : private std::conditional_t<intrusive, intrusive_ptr_target, Empty> {
 public:
  // Creates an instance from an existing storage generation.
  explicit ShadowStorageImpl(std::uint64_t generation) noexcept;

  // Gets the current generation.
  auto generation() const noexcept -> std::uint64_t;

  // Increments the current generation.
  auto bump_generation() noexcept -> std::uint64_t;

 private:
  // From the user's perspective, copies are fully distinct and
  // require no external synchronization, so we need to ensure this
  // field's concurrency is properly managed.
  std::atomic<std::uint64_t> generation_;

  friend intrusive_ptr<ShadowStorageImpl</*intrusive=*/true>>;
};

} // namespace c10::impl::cow::detail

namespace c10::impl::cow {

using ShadowStorage = detail::ShadowStorageImpl<true>;
using ShadowStorageNonIntrusive = detail::ShadowStorageImpl<false>;

} // namespace c10::impl::cow
