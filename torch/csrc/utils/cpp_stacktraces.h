#pragma once

#include <torch/csrc/Export.h>

namespace torch {
TORCH_API bool get_cpp_stacktraces_enabled();
TORCH_API bool get_disable_addr2line();
} // namespace torch
