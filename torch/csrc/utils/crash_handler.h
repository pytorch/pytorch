#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <string>

namespace torch {
namespace crash_handler {

#ifdef _WIN32
typedef std::wstring STRING_TYPE;
#else
typedef std::string STRING_TYPE;
#endif

// Set up a handler that writes minidumps to 'dir' on signals. This is not
// necessary to call unless you want to change 'dir' to something other than
// the default '/tmp/pytorch_crashes'.
TORCH_API void enable_minidumps(const STRING_TYPE& dir);

// Enable minidumps when passing exceptions up to Python. By default these don't
// do anything special, but it can be useful to write out a minidump on
// exceptions for debugging purposes. This has no effect in C++.
TORCH_API void enable_minidumps_on_exceptions();

// Disable all minidump writing and un-register the signal handler
TORCH_API void disable_minidumps();

// Get the directory that minidumps will be written to
TORCH_API const STRING_TYPE& get_minidump_directory();

// These are TORCH_API'ed since they are used from libtorch_python.so
TORCH_API bool is_enabled_on_exceptions();
TORCH_API void write_minidump();

} // namespace crash_handler
} // namespace torch
