#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>

bool my_is_warn_always_enabled() {
  return torch_is_warn_always_enabled();
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_is_warn_always_enabled() -> bool");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_is_warn_always_enabled", TORCH_BOX(&my_is_warn_always_enabled));
}
