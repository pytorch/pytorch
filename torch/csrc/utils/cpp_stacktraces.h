#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/profiler/unwind/unwind.h>

namespace torch {
TORCH_API bool get_cpp_stacktraces_enabled();
TORCH_API torch::unwind::Mode get_symbolize_mode();
} // namespace torch
