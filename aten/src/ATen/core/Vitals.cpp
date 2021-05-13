#include <ATen/core/Vitals.h>
#include <cstdlib>


namespace at {
namespace vitals {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
APIVitals VitalsAPI;

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

bool APIVitals::setVital(
    const std::string& vital_name,
    const std::string& attr_name,
    const std::string& value) {
  if (!torchVitalEnabled()) {
    return false;
  }

  auto iter = name_map_.find(vital_name);
  TorchVital *vital = nullptr;
  if (iter == name_map_.end()) {
    auto r = name_map_.emplace(std::make_pair(vital_name, TorchVital(vital_name)));
    vital = &r.first->second;
  } else {
    vital = &iter->second;
  }

  vital->create(attr_name) << value;
  return true;
}

} // namespace at
} // namespace vitals
