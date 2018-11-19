#pragma once
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/jit/ir.h"
#include <iostream>
#include <vector>


namespace torch { namespace jit {
TORCH_API std::vector<at::Tensor> PythonPrint(std::ostream& out, const Graph& graph, bool enforce_importable=false);
TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
}}
