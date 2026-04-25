#pragma once
#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <string>

namespace torch::csrc::shim::details {
/// Store an exception and its backtrace that occurred in the calling thread.
AOTI_TORCH_EXPORT void set_torch_exception_what(const std::string& our_message);

/// Store an exception that occurred in the calling thread.
AOTI_TORCH_EXPORT void set_torch_exception_what_without_backtrace(
    const std::string& our_message);

/// Retrieves the exception and backtrace that was stored in this thread.
/// This reference is only valid while the calling thread exists and no other
/// exception is stored for the thread.
AOTI_TORCH_EXPORT const std::string& get_torch_exception_what();

/// Retrieves the exception that was stored in this thread.
/// This reference is only valid while the calling thread exists and no other
/// exception is stored for the thread.
AOTI_TORCH_EXPORT const std::string&
get_torch_exception_what_without_backtrace();

/// Configures the backtrace printing state for the calling thread. Returns the
/// previous state.
AOTI_TORCH_EXPORT bool torch_exception_state_set_exception_printing(
    const bool should_print);

/// Retrieves the backtrace printing state for this thread.
AOTI_TORCH_EXPORT bool torch_exception_state_get_exception_printing();
} // namespace torch::csrc::shim::details
