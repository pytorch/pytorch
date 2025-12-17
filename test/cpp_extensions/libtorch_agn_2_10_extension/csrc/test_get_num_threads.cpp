#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>

uint32_t test_get_num_threads() {
  return torch::stable::get_num_threads();
}

STABLE_TORCH_LIBRARY_FRAGMENT(libtorch_agn_2_10, m) {
  m.def("test_get_num_threads() -> int");
}

STABLE_TORCH_LIBRARY_IMPL(libtorch_agn_2_10, CompositeExplicitAutograd, m) {
  m.impl("test_get_num_threads", TORCH_BOX(&test_get_num_threads));
}
