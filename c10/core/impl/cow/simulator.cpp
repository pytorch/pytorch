#include <c10/core/impl/cow/simulator.h>

#include <c10/util/Exception.h>

#include <limits>

namespace c10::impl {

template <bool intrusive>
cow::SimulatorImpl<intrusive>::SimulatorImpl(
    std::uint64_t storage_generation) noexcept
    : storage_generation_(storage_generation) {}

template <bool intrusive>
auto cow::SimulatorImpl<intrusive>::storage_generation() noexcept
    -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  return storage_generation_;
}

template <bool intrusive>
auto cow::SimulatorImpl<intrusive>::bump_storage_generation() noexcept
    -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  TORCH_INTERNAL_ASSERT(
      storage_generation_ != std::numeric_limits<std::uint64_t>::max());
  return ++storage_generation_;
}

template class cow::SimulatorImpl<true>;
template class cow::SimulatorImpl<false>;

} // namespace c10::impl
