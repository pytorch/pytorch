#include <torch/csrc/stable/cpu/Utils.h>
#include <torch/csrc/stable/library.h>

#include <cstdint>

int64_t my_l1d_cache_size() {
  return torch::stable::cpu::get_l1d_cache_size();
}

int64_t my_l2_cache_size() {
  return torch::stable::cpu::get_l2_cache_size();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_l1d_cache_size() -> int");
  m.def("my_l2_cache_size() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_l1d_cache_size", TORCH_BOX(&my_l1d_cache_size));
  m.impl("my_l2_cache_size", TORCH_BOX(&my_l2_cache_size));
}
