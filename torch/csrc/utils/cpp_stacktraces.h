#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch {
TORCH_API bool get_cpp_stacktraces_enabled();
// Overrides the cached cpp-stacktraces-enabled value. Used so that other
// subsystems (e.g. c10d debug level) can force stacktraces on even after
// the env-var-derived value has already been cached.
TORCH_API void set_cpp_stacktraces_enabled(bool enabled);
TORCH_API torch::unwind::Mode get_symbolize_mode();
} // namespace torch
