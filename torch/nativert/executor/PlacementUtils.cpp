#include <ATen/DeviceAccelerator.h>
#include <torch/nativert/executor/Placement.h>

namespace torch::nativert {

bool isSameDevice(const c10::Device& a, const c10::Device& b) {
  if (a.type() != b.type()) {
    return false;
  }
  if (!at::accelerator::isAccelerator(a.type())) {
    return true;
  }
  auto resolveIndex = [](const c10::Device& d) -> c10::DeviceIndex {
    if (d.has_index()) {
      return d.index();
    }
    return at::accelerator::getDeviceIndex();
  };
  return resolveIndex(a) == resolveIndex(b);
}
} // namespace torch::nativert
