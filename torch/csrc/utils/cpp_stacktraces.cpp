#include <torch/csrc/utils/cpp_stacktraces.h>

#include <c10/util/Exception.h>
#include <c10/util/env.h>

namespace torch {
namespace {
bool compute_cpp_stack_traces_enabled() {
  return c10::utils::check_env("TORCH_SHOW_CPP_STACKTRACES") == true;
}

} // namespace

bool get_cpp_stacktraces_enabled() {
  static bool enabled = compute_cpp_stack_traces_enabled();
  return enabled;
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
  }
  // The in-process symbolizer is the default. TORCH_DISABLE_ADDR2LINE is kept
  // for backwards compatibility: =1 falls back to dladdr, =0 opts into the
  // external addr2line process (previously the default).
  auto disable_addr2line = c10::utils::check_env("TORCH_DISABLE_ADDR2LINE");
  if (disable_addr2line == true) {
    return unwind::Mode::dladdr;
  } else if (disable_addr2line == false) {
    return unwind::Mode::addr2line;
  }
  return unwind::Mode::fast;
}

unwind::Mode get_symbolize_mode() {
  static unwind::Mode mode = compute_symbolize_mode();
  return mode;
}

} // namespace torch
