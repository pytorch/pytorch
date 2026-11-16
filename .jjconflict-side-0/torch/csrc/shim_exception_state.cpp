#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <torch/csrc/shim_exception_state.h>
#include <string>

namespace torch::csrc::shim::details {
// Thread local storage for the most recent exception's message and backtrace.
thread_local std::string torch_exception_what;

// Thread local storage for the most recent exception's message.
thread_local std::string torch_exception_what_without_backtrace;

AOTI_TORCH_EXPORT void set_torch_exception_what(
    const std::string& our_message) {
  torch_exception_what = our_message;
}

AOTI_TORCH_EXPORT void set_torch_exception_what_without_backtrace(

    const std::string& our_message) {
  torch_exception_what_without_backtrace = our_message;
}

AOTI_TORCH_EXPORT const std::string& get_torch_exception_what() {
  return torch_exception_what;
}

AOTI_TORCH_EXPORT const std::string&
get_torch_exception_what_without_backtrace() {
  return torch_exception_what_without_backtrace;
}
} // namespace torch::csrc::shim::details
