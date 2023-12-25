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

} // namespace xpu
} // namespace torch
