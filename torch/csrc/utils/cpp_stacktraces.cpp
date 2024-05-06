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

bool get_disable_addr2line() {
  static bool disabled = compute_disable_addr2line();
  return disabled;
}

} // namespace torch
