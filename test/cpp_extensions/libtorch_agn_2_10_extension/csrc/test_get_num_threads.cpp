#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

uint32_t test_get_num_threads() {
  return torch::stable::get_num_threads();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_get_num_threads() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("test_get_num_threads", TORCH_BOX(&test_get_num_threads));
}
