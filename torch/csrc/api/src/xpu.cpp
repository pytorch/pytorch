#include <ATen/Context.h>
#include <torch/xpu.h>

namespace torch::xpu {

size_t device_count() {
  return at::detail::getXPUHooks().deviceCount();
}

bool is_available() {
  return xpu::device_count() > 0;
}

void manual_seed(uint64_t seed) {
  if (is_available()) {
    auto index = at::detail::getXPUHooks().getCurrentDevice();
    auto gen = at::detail::getXPUHooks().getDefaultXPUGenerator(index);
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

/// Sets the seed for all available GPUs.
void manual_seed_all(uint64_t seed) {
  auto num_gpu = device_count();
  for (const auto i : c10::irange(num_gpu)) {
    auto gen = at::detail::getXPUHooks().getDefaultXPUGenerator(
        static_cast<c10::DeviceIndex>(i));
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

void synchronize(int64_t device_index) {
  TORCH_CHECK(is_available(), "No XPU are available");
  at::detail::getXPUHooks().deviceSynchronize(
      static_cast<c10::DeviceIndex>(device_index));
}

} // namespace torch::xpu
