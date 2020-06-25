#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

void InitializerPeepholeONNX(
    Block* b,
    std::map<std::string, IValue>& paramDict);

} // namespace jit

} // namespace torch
