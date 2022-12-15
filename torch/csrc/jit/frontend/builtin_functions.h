#pragma once

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

TORCH_API const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name);
} // namespace jit
} // namespace torch
