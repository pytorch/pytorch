#include <torch/xpu.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
namespace xpu {

size_t device_count() {
  return at::detail::getXPUHooks().getNumGPUs();
}

bool is_available() {
  return xpu::device_count() > 0;
}

void synchronize(int64_t device_index) {
  TORCH_CHECK(is_available(), "No XPU are available");
  at::detail::getXPUHooks().deviceSynchronize(
      static_cast<c10::DeviceIndex>(device_index));
}

} // namespace xpu
} // namespace torch
