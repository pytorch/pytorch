#include <ATen/core/Vitals.h>
#include <cstdlib>
#include <iostream>

namespace at::vitals {

APIVitals VitalsAPI;

std::ostream& operator<<(std::ostream& os, TorchVital const& tv) {
  for (const auto& m : tv.attrs) {
    os << "[TORCH_VITAL] " << tv.name << "." << m.first << "\t\t "
       << m.second.value << "\n";
  }
  return os;
}

TorchVital::~TorchVital() {
  if (torchVitalEnabled()) {
    std::cout << *this;
  }
}

TorchVitalAttr& TorchVital::create(const std::string& attr) {
  return create(attr, /* force = */ false);
}

TorchVitalAttr& TorchVital::create(const std::string& attr, bool force) {
  if (!(torchVitalEnabled() || force)) {
    static TorchVitalAttr disabled;
    return disabled;
  }
  auto iter = attrs.find(attr);
  if (iter == attrs.end()) {
    auto r = attrs.emplace(attr, TorchVitalAttr());
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
      return e[0] != '\0';
    }
    return false;
  }();
  if (enabled) {
    VitalsAPI.vitals_enabled = true;
  }
  return VitalsAPI.vitals_enabled;
}

std::string APIVitals::readVitals() {
  if (!torchVitalEnabled()) {
    return "";
  }

  std::stringstream buf;
  for (const auto& x : name_map_) {
    buf << x.second;
  }
  return buf.str();
}

bool APIVitals::setVital(
    const std::string& vital_name,
    const std::string& attr_name,
    const std::string& value,
    bool force) {
  if (!(torchVitalEnabled() || force)) {
    return false;
  }

  auto iter = name_map_.find(vital_name);
  TorchVital* vital = nullptr;
  if (iter == name_map_.end()) {
    auto r = name_map_.emplace(vital_name, TorchVital(vital_name));
    vital = &r.first->second;
  } else {
    vital = &iter->second;
  }

  vital->create(attr_name, force).write(value, force);
  return true;
}

APIVitals::APIVitals() : vitals_enabled(false), name_map_() {
  // Set default values, force is necessary because in unit tests the env
  // variable may not be set when global APIVitals are constructed.
  setVital("CUDA", "used", "False", /* force = */ true);
}

} // namespace at::vitals
