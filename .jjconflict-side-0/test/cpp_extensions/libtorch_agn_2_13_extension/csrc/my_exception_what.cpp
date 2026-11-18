#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/macros.h>

#include <string>

// Direct calls to the 2.13 exception getter shims. Also tested in
// our_subtract_stable_error_check (in the 2.10 extension, which reaches these
// shims dynamically via TORCH_DYNAMIC_VERSION_CALL when built at an older
// target).
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
  m.impl("my_exception_get_what_without_backtrace", TORCH_BOX(&my_exception_get_what_without_backtrace));
}
