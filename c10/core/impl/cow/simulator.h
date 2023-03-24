#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <mutex>
#include <type_traits>

namespace c10::impl::cow {

namespace detail {

// An empty struct to use as the base class for a non-intrusive
// SimulatorImpl.
struct Empty {};

} // namespace detail

// Simulates a copy-on-write storage for a tensor.
//
// This simulator is used to identify situations where a copy-on-write
// tensor would have been created and whether or not a logical copy
// uses the storage after a different copy has modified it. So in
// theory, it can identify reads/writes to copies after a write to a
// different copy that is presently an alias.
//
// However, the fidelity of this check is limited by the extent to
// which we have instrumented operations as reading or writing to
// storage.
//
// We use a monotonically increasing generation number to track
// modifications to storage.
template <bool intrusive = true>
class C10_API SimulatorImpl final
    : private std::
          conditional_t<intrusive, intrusive_ptr_target, detail::Empty> {
 public:
  // Creates an instance from an existing storage generation.
  explicit SimulatorImpl(std::uint64_t storage_generation) noexcept;

  // Gets the current generation.
  auto storage_generation() noexcept -> std::uint64_t;

  // Increments the current generation.
  auto bump_storage_generation() noexcept -> std::uint64_t;

 private:
  std::mutex mtx_;
  // From the user's perspective, copies are fully distinct and
  // require no external synchronization, so we need to ensure this
  // field's concurrency is properly managed.
  std::uint64_t storage_generation_;

  friend intrusive_ptr<SimulatorImpl</*intrusive=*/true>>;
};

using Simulator = SimulatorImpl<true>;

} // namespace c10::impl::cow
