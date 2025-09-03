#include <torch/nativert/executor/Placement.h>

#include <fmt/ostream.h>

namespace torch::nativert {

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
