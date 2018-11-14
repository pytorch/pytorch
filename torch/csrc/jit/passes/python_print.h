#pragma once
#include "torch/csrc/WindowsTorchApiMacro.h"
#include <iostream>

namespace c10 {
  struct Symbol;
}

namespace torch { namespace jit {
struct Graph;
TORCH_API std::ostream& PythonPrint(std::ostream& out, const Graph& graph);
TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
}}
