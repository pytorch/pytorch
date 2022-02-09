#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// This pass creates the subgraphs for oneDNN Graph Fusion Nodes.
// Its code-structure has been vastly inspired from
// torch/csrc/jit/passes/create_autodiff_subgraphs.cpp
void CreateLlgaSubgraphs(std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
