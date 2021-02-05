#pragma once
#include <cstring>
#include <iostream>
#include <sstream>
#include <unordered_map>

namespace at {
namespace vitals {

bool torchVitalEnabled();

struct TorchVitalAttr {
  // always initialized to empty
  std::string value = "";
  template <typename T>
  TorchVitalAttr& operator<<(const T& t) {
    if (torchVitalEnabled()) {
      std::stringstream ss;
      ss << t;
      value += ss.str();
    }
    return *this;
  }
};

struct TorchVital {
  std::string name;
  std::unordered_map<std::string, TorchVitalAttr> attrs;

  explicit TorchVital(std::string n) : name(std::move(n)) {}
  TorchVital() = delete;

  TorchVitalAttr& create(const std::string& attr);

  ~TorchVital() {
    for (const auto& m : attrs) {
      std::cout << "[TORCH_VITAL] " << name << "." << m.first << "\t\t "
                << m.second.value << "\n";
    }
  }
};

} // namespace at
} // namespace vitals

#define TORCH_VITAL_DECLARE(name) extern TorchVital TorchVital_##name;

#define TORCH_VITAL_DEFINE(name) TorchVital TorchVital_##name(#name);

#define TORCH_VITAL(name, attr) TorchVital_##name.create(#attr)
