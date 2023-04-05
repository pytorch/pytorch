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
  TORCH_CHECK(is_available(), "No MPS devices are available");
  at::detail::getMPSHooks().deviceSynchronize();
}

} // namespace mps
} // namespace torch
