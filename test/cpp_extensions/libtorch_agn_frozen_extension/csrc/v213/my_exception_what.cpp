// 2.13-only ops: compiled only when TORCH_TARGET_VERSION >= 2.13.

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

#include <string>

std::string my_exception_what() {
  return std::string(torch_exception_get_what());
}

std::string my_exception_get_what_without_backtrace() {
  return std::string(torch_exception_get_what_without_backtrace());
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_exception_what() -> str");
  m.def("my_exception_get_what_without_backtrace() -> str");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_exception_what", TORCH_BOX(&my_exception_what));
  m.impl(
      "my_exception_get_what_without_backtrace",
      TORCH_BOX(&my_exception_get_what_without_backtrace));
}
