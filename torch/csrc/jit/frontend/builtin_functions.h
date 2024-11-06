#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>

namespace torch::jit {

TORCH_API const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name);
} // namespace torch::jit
