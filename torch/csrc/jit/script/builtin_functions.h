#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {
namespace script {

TORCH_API const std::vector<Function*>& getAllBuiltinFunctionsFor(Symbol name);
}
} // namespace jit
} // namespace torch
