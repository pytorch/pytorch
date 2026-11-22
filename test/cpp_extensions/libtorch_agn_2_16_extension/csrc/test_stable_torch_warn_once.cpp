#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

void test_stable_torch_warn_once(int64_t value) {
  if (value < 0) {
    STABLE_TORCH_WARN_ONCE("test_stable_torch_warn_once negative value");
  } else {
    STABLE_TORCH_WARN_ONCE("test_stable_torch_warn_once value=", value);
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_stable_torch_warn_once(int value) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl(
      "test_stable_torch_warn_once", TORCH_BOX(&test_stable_torch_warn_once));
}
