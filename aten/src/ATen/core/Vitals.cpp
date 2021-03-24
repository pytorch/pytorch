#include <ATen/core/Vitals.h>
#include <cstdlib>

namespace at {
namespace vitals {

TorchVitalAttr& TorchVital::create(const std::string& attr) {
  if (!torchVitalEnabled()) {
    static TorchVitalAttr disabled;
    return disabled;
  }
  auto iter = attrs.find(attr);
  if (iter == attrs.end()) {
    auto r = attrs.emplace(std::make_pair(attr, TorchVitalAttr()));
    return r.first->second;
  }
  return iter->second;
}

bool torchVitalEnabled() {
  // If this is a performance hit, make `enabled` variable static
  // and return `const bool&` instead
  bool enabled = []() {
    auto e = getenv("TORCH_VITAL");
    if (e != nullptr) {
      return strlen(e) > 0;
    }
    return false;
  }();
  return enabled;
}

} // namespace at
} // namespace vitals
