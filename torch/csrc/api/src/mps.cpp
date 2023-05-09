#include <torch/mps.h>

#include <ATen/Context.h>
#include <c10/util/irange.h>

#include <cstddef>

namespace torch {
namespace mps {

bool is_available() {
  return at::detail::getMPSHooks().hasMPS();
}

/// Sets the seed for the MPS's default generator.
void manual_seed(uint64_t seed) {
  if (is_available()) {
    auto gen = at::detail::getMPSHooks().getDefaultMPSGenerator();
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

void synchronize() {
  at::detail::getMPSHooks().deviceSynchronize();
}

void commit() {
  at::detail::getMPSHooks().commitStream();
}

MTLCommandBuffer_t get_command_buffer() {
  return static_cast<MTLCommandBuffer_t>(
      at::detail::getMPSHooks().getCommandBuffer());
}

dispatch_queue_t get_dispatch_queue() {
  return static_cast<dispatch_queue_t>(
      at::detail::getMPSHooks().getDispatchQueue());
}

} // namespace mps
} // namespace torch
