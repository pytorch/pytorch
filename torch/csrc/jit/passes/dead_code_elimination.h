#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch { namespace jit {

// If given a top-level graph, DCE will construct do alias analysis that allows
// for "smarter" dead code elimination (we will eliminate mutable ops if we can
// prove the mutated values are not used). Otherwise, we will not allow DCE to
// eliminate mutable ops.
//
// So, prefer to use the graph version if you can.
TORCH_API void EliminateDeadCode(const std::shared_ptr<Graph>& graph);
TORCH_API void EliminateDeadCode(Block *block, bool recurse=true);

TORCH_API std::unordered_set<Node*> FindDeadNodes(Block *block, bool recurse=true);

}}
