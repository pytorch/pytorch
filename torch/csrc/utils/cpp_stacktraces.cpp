#include <torch/csrc/utils/cpp_stacktraces.h>

#include <cstdlib>
#include <cstring>

#include <c10/util/Exception.h>

namespace torch {
namespace {
bool compute_cpp_stack_traces_enabled() {
  auto envar = std::getenv("TORCH_SHOW_CPP_STACKTRACES");
  if (envar) {
    if (strcmp(envar, "0") == 0) {
      return false;
    }
    if (strcmp(envar, "1") == 0) {
      return true;
    }
    TORCH_WARN(
        "ignoring invalid value for TORCH_SHOW_CPP_STACKTRACES: ",
        envar,
        " valid values are 0 or 1.");
  }
  return false;
}

bool compute_disable_addr2line() {
  auto envar = std::getenv("TORCH_DISABLE_ADDR2LINE");
  if (envar) {
    if (strcmp(envar, "0") == 0) {
      return false;
    }
    if (strcmp(envar, "1") == 0) {
      return true;
    }
    TORCH_WARN(
        "ignoring invalid value for TORCH_DISABLE_ADDR2LINE: ",
        envar,
        " valid values are 0 or 1.");
  }
  return false;
}
} // namespace

bool get_cpp_stacktraces_enabled() {
  static bool enabled = compute_cpp_stack_traces_enabled();
  return enabled;
}

static torch::unwind::Mode compute_symbolize_mode() {
  auto envar_c = std::getenv("TORCH_SYMBOLIZE_MODE");
  if (envar_c) {
    std::string envar = envar_c;
    if (envar == "dladdr") {
      return unwind::Mode::dladdr;
    } else if (envar == "addr2line") {
      return unwind::Mode::addr2line;
    } else if (envar == "fast") {
      return unwind::Mode::fast;
    } else {
      TORCH_CHECK(
          false,
          "expected {dladdr, addr2line, fast} for TORCH_SYMBOLIZE_MODE, got ",
          envar);
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
