#include <torch/csrc/stable/cpu/utils.h>
#include <torch/csrc/stable/library.h>

#include <cstdint>
#include <optional>
#include <string>

std::optional<int64_t> my_cpu_capability_int(
    const std::string& key,
    std::optional<int64_t> fallback) {
  if (fallback) {
    return torch::stable::cpu::get_cpu_capability<int64_t>(
        key.c_str(), *fallback);
  }
  return torch::stable::cpu::get_cpu_capability<int64_t>(key.c_str());
}

std::optional<bool> my_cpu_capability_bool(
    const std::string& key,
    std::optional<bool> fallback) {
  if (fallback) {
    return torch::stable::cpu::get_cpu_capability<bool>(
        key.c_str(), *fallback);
  }
  return torch::stable::cpu::get_cpu_capability<bool>(key.c_str());
}

std::optional<std::string> my_cpu_capability_string(
    const std::string& key,
    std::optional<std::string> fallback) {
  if (fallback) {
    return torch::stable::cpu::get_cpu_capability<std::string>(
        key.c_str(), *fallback);
  }
  return torch::stable::cpu::get_cpu_capability<std::string>(key.c_str());
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_cpu_capability_int(str key, int? fallback=None) -> int?");
  m.def("my_cpu_capability_bool(str key, bool? fallback=None) -> bool?");
  m.def("my_cpu_capability_string(str key, str? fallback=None) -> str?");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_cpu_capability_int", TORCH_BOX(&my_cpu_capability_int));
  m.impl("my_cpu_capability_bool", TORCH_BOX(&my_cpu_capability_bool));
  m.impl("my_cpu_capability_string", TORCH_BOX(&my_cpu_capability_string));
}
