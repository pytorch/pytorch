#pragma once
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <map>

#include <c10/core/impl/LocalDispatchKeySet.h>

namespace at {
namespace vitals {

TORCH_API bool torchVitalEnabled();

struct TORCH_API TorchVitalAttr {
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

struct TORCH_API TorchVital {
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

// A way to access vitals by string names instead of by global reference.
// This enables access to vitals from the PythonAPI.
class TORCH_API APIVitals
{
public:
    // Set any vital sign that was added to the map.
    bool setVital(const std::string &vital_name, const std::string &attr_name, const std::string &value);

    APIVitals(): name_map_() { }

    // Ensure this stays a singleton
    APIVitals(APIVitals const& other) = delete;
    APIVitals(APIVitals&& other) = delete;
    APIVitals& operator=(const APIVitals&) = delete;
    APIVitals& operator=(APIVitals&&) = delete;

private:
    std::unordered_map<std::string, TorchVital> name_map_;
};

extern TORCH_API APIVitals VitalsAPI;

} // namespace at
} // namespace vitals

#define TORCH_VITAL_DECLARE(name) TORCH_API at::vitals::TorchVital TorchVital_##name;

#define TORCH_VITAL_DEFINE(name) TORCH_API at::vitals::TorchVital TorchVital_##name(#name);

#define TORCH_VITAL_BASE(name) TorchVital_##name

#define TORCH_VITAL(name, attr) TORCH_VITAL_BASE(name).create(#attr)
