#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

TORCH_API void EliminateDeadCode(const std::shared_ptr<Graph>& graph);
TORCH_API void EliminateDeadCode(Block *block, bool recurse=true);

}}
