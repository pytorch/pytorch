#include <ATen/Context.h>
#include <torch/xpu.h>

namespace torch::xpu {

size_t device_count() {
  return at::detail::getXPUHooks().getNumGPUs();
}

bool is_available() {
  return xpu::device_count() > 0;
}

} // namespace torch::xpu
