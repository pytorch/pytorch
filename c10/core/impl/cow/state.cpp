#include <c10/core/impl/cow/state.h>

#include <c10/util/Exception.h>

namespace c10::impl {

auto cow::State::maybe_bump(cow::Simulator* maybe_simulator) -> void {
  std::lock_guard<std::mutex> lock(mtx_);
  if (!default_simulator_.has_value()) {
    // Any created simulator should be bound to the storage that it
    // was created from. Hence, there should only be an external
    // simulator if the storage created it. We don't check for the
    // specific matching of the storage and simulator, but we do check
    // that the presence relationship holds.
    //
    // TODO: This check should probably be here, but it is actually
    // triggering in the StaticRuntime.autogen_inner test.
    //
    // TORCH_INTERNAL_ASSERT(maybe_simulator == nullptr);
    return;
  }

  std::uint64_t physical_generation = ++physical_generation_;
  std::uint64_t shadow_generation = maybe_simulator != nullptr
      ? maybe_simulator->bump_storage_generation()
      : default_simulator_->bump_storage_generation();

  TORCH_INTERNAL_ASSERT(shadow_generation <= physical_generation);
  if (shadow_generation != physical_generation) {
    TORCH_WARN_ONCE(
        "You have written through to both aliases created by calling "
        "reshape(). In the future, reshape() will never create a view but will "
        "instead return a lazily copied tensor. If you wish to preserve the "
        "aliasing properties, you should rewrite your reshape() as a view().");
  }
}

auto cow::State::storage_generation() -> std::uint64_t {
  std::lock_guard<std::mutex> lock(mtx_);
  return physical_generation_;
}

auto cow::State::simulate_copy_on_write(cow::Simulator* simulator)
    -> intrusive_ptr<cow::Simulator> {
  std::lock_guard<std::mutex> lock(mtx_);
  if (simulator != nullptr) {
    TORCH_INTERNAL_ASSERT(default_simulator_.has_value());
    return make_intrusive<cow::Simulator>(simulator->storage_generation());
  }

  if (!default_simulator_.has_value()) {
    TORCH_INTERNAL_ASSERT(physical_generation_ == 0);
    default_simulator_.emplace(physical_generation_);
  }

  return make_intrusive<cow::Simulator>(physical_generation_);
}

} // namespace c10::impl
