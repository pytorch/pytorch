#include <torch/nativert/executor/Placement.h>

#include <fmt/ostream.h>

namespace torch::nativert {

c10::Device normalizeDevice(const c10::Device& device) {
  // cpu device doesn't have index
  // cuda device index must have a index
  if (device.is_cpu()) {
    return c10::Device(c10::DeviceType::CPU);
  } else if (device.is_cuda()) {
    return c10::Device(
        c10::DeviceType::CUDA,
        device.has_index() ? device.index() : static_cast<c10::DeviceIndex>(0));
  } else {
    TORCH_CHECK(false, "Unsupported device type", device);
  }
}

bool isSameDevice(const c10::Device& a, const c10::Device& b) {
  if (a.is_cpu()) {
    return b.is_cpu();
  }
  if (a.is_cuda()) {
    if (b.is_cuda()) {
      auto aIndex = a.has_index() ? a.index() : 0;
      auto bIndex = b.has_index() ? b.index() : 0;
      return aIndex == bIndex;
    } else {
      return false;
    }
  }
  TORCH_CHECK(false, "Unsupported device type", a, " and ", b);
  return false;
}
} // namespace torch::nativert
