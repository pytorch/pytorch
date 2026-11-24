#include <torch/csrc/stable/cpu.h>
#include <torch/csrc/stable/library.h>

#include <cstdint>
#include <optional>
#include <string>

template <typename T>
std::optional<T> my_cpu_capability(const std::string& key) {
  return torch::stable::cpu::get_cpu_capability<T>(key.c_str());
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_cpu_capability_int(str key) -> int?");
  m.def("my_cpu_capability_bool(str key) -> bool?");
  m.def("my_cpu_capability_string(str key) -> str?");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_cpu_capability_int", TORCH_BOX(&my_cpu_capability<int64_t>));
  m.impl("my_cpu_capability_bool", TORCH_BOX(&my_cpu_capability<bool>));
  m.impl("my_cpu_capability_string", TORCH_BOX(&my_cpu_capability<std::string>));
}
