#include <torch/csrc/utils/cpp_stacktraces.h>

#include <c10/util/Exception.h>
#include <c10/util/env.h>

namespace torch {
namespace {
bool compute_cpp_stack_traces_enabled() {
  return c10::utils::check_env("TORCH_SHOW_CPP_STACKTRACES") == true;
}

bool compute_disable_addr2line() {
  return c10::utils::check_env("TORCH_DISABLE_ADDR2LINE") == true;
}

// Owns the single cached flag so both the getter and setter operate on the
// same storage. get_cpp_stacktraces_enabled() previously computed this via
// its own local static, which meant there was no way to override it later
// (e.g. from c10d's debug level) -- returning by value gave callers a copy,
// not the underlying storage.
bool& cpp_stacktraces_enabled_ref() {
  static bool enabled = compute_cpp_stack_traces_enabled();
  return enabled;
}
} // namespace

bool get_cpp_stacktraces_enabled() {
  return cpp_stacktraces_enabled_ref();
}

void set_cpp_stacktraces_enabled(bool enabled) {
  cpp_stacktraces_enabled_ref() = enabled;
}

static torch::unwind::Mode compute_symbolize_mode() {
  auto envar_c = c10::utils::get_env("TORCH_SYMBOLIZE_MODE");
  if (envar_c.has_value()) {
    if (envar_c == "dladdr") {
      return unwind::Mode::dladdr;
    } else if (envar_c == "addr2line") {
      return unwind::Mode::addr2line;
    } else if (envar_c == "fast") {
      return unwind::Mode::fast;
    } else {
      TORCH_CHECK(
          false,
          "expected {dladdr, addr2line, fast} for TORCH_SYMBOLIZE_MODE, got ",
          envar_c.value());
    }
  } else {
    return compute_disable_addr2line() ? unwind::Mode::dladdr
                                       : unwind::Mode::addr2line;
  }
}

unwind::Mode get_symbolize_mode() {
  static unwind::Mode mode = compute_symbolize_mode();
  return mode;
}

} // namespace torch
