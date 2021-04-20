#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <string>


namespace torch {
namespace crash_handler {

TORCH_API void _enable_minidump_collection(const std::string& dir);
TORCH_API void _disable_minidump_collection();

TORCH_API const std::string& _get_minidump_directory();

bool is_enabled();

void write_minidump();

} // namespace crash_handler
} // namepsace torch
