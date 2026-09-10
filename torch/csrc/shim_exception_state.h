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
} // namespace torch::csrc::shim::details
