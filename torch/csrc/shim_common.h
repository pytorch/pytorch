#pragma once
#include <torch/csrc/inductor/aoti_torch/c/macros.h>
#include <string>

// Thread local variable that holds the most recent exception message with
// backtrace.
AOTI_TORCH_EXPORT extern thread_local std::string torch_exception_what;

// Thread local variable that holds the most recent exception message.
AOTI_TORCH_EXPORT extern thread_local std::string
    torch_exception_what_without_backtrace;

/// Thread local variable that specifies if the exception and its backtrace
/// is printed when an exception occurs. Defaults to true.
AOTI_TORCH_EXPORT extern thread_local bool torch_exception_printing_enabled;
