#include <torch/csrc/utils/cpp_stacktraces.h>

#include <c10/util/Exception.h>
#include <c10/util/env.h>

#include <algorithm>
#include <cctype>
#include <string>

namespace torch {
namespace {
bool compute_cpp_stack_traces_enabled() {
  auto show_cpp_stacktraces =
      c10::utils::check_env("TORCH_SHOW_CPP_STACKTRACES");
  if (show_cpp_stacktraces.has_value()) {
    return show_cpp_stacktraces.value();
  }

  auto distributed_debug =
      c10::utils::get_env("TORCH_DISTRIBUTED_DEBUG").value_or("");
  std::transform(
      distributed_debug.begin(),
      distributed_debug.end(),
      distributed_debug.begin(),
      [](unsigned char c) { return std::toupper(c); });
  return distributed_debug == "DETAIL";
}

bool compute_disable_addr2line() {
  return c10::utils::check_env("TORCH_DISABLE_ADDR2LINE").value_or(true);
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
