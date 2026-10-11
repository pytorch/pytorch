#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

void test_std_torch_warn(int64_t value) {
  STD_TORCH_WARN(
      "test_std_torch_warn value=", value, " scale=", 0.5, ' ', true);
}

void test_std_torch_warn_once(int64_t value) {
  if (value < 0) {
    STD_TORCH_WARN_ONCE("test_std_torch_warn_once negative value");
  } else {
    STD_TORCH_WARN_ONCE("test_std_torch_warn_once value=", value);
  }
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("test_std_torch_warn(int value) -> ()");
  m.def("test_std_torch_warn_once(int value) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("test_std_torch_warn", TORCH_BOX(&test_std_torch_warn));
  m.impl("test_std_torch_warn_once", TORCH_BOX(&test_std_torch_warn_once));
}
