
#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {

void validateIr(Fusion* fusion);

} // namespace fuser
} // namespace jit
} // namespace torch
