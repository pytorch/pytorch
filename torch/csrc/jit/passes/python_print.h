#pragma once
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/jit/ir.h"
#include <iostream>
#include <vector>


namespace torch { namespace jit {

namespace script {
  struct Method;
  struct Module;
}

TORCH_API std::vector<at::Tensor> PythonPrint(std::ostream& out, Graph& graph, bool enforce_importable=false);
TORCH_API std::vector<at::Tensor> PythonPrint(std::ostream& out, script::Method& graph, bool enforce_importable=false);
TORCH_API std::vector<at::Tensor> PythonPrint(std::ostream& out, script::Module& module, bool enforce_importable=false);

TORCH_API bool printerHasSpecialCaseFor(c10::Symbol sym);
}}
