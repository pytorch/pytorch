#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <torch/csrc/shim_exception_state.h>
#include <string>

namespace torch::csrc::shim::details {
// Thread local storage for the most recent exception's message and backtrace.
thread_local std::string torch_exception_what;

// Thread local storage for the most recent exception's message.
thread_local std::string torch_exception_what_without_backtrace;

// Default to printing the exception since that was the historical behaviour.
thread_local bool torch_exception_printing_enabled = true;

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

AOTI_TORCH_EXPORT bool torch_exception_state_set_exception_printing(
    const bool should_print) {
  const bool previous = torch_exception_printing_enabled;
  torch_exception_printing_enabled = should_print;
  return previous;
}

AOTI_TORCH_EXPORT bool torch_exception_state_get_exception_printing() {
  return torch_exception_printing_enabled;
}
} // namespace torch::csrc::shim::details
