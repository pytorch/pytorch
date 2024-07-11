#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API std::pair<Module, std::vector<IValue>> list_module_parameters(
    const Module& module);

} // namespace jit
} // namespace torch
