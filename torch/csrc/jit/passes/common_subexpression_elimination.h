#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

TORCH_API void EliminateCommonSubexpression(std::shared_ptr<Graph>& graph);
TORCH_API void EliminateCommonSubexpression(Block * block,
    std::function<Node*(Node*)> parent_lookup_fn = [](Node*){ return nullptr; });

}}
