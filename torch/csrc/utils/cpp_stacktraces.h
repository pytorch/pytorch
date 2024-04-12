#pragma once

#include <torch/csrc/Export.h>

namespace torch {
TORCH_API bool get_cpp_stacktraces_enabled();
enum class SymbolizeMode {
  dladdr,
  addr2line,
  fast,
};
TORCH_API SymbolizeMode get_symbolize_mode();
} // namespace torch
