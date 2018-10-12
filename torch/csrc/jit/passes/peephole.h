#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

TORCH_API void PeepholeOptimize(std::shared_ptr<Graph>& graph, bool addmm_fusion_enabled=false);

}}
