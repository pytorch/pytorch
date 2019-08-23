#pragma once

#include <torch/csrc/jit/ir.h>

namespace torch {
namespace jit {

// If given a top-level graph, DCE will construct do alias analysis that allows
// for "smarter" dead code elimination (we will eliminate mutable ops if we can
// prove the mutated values are not used). Otherwise, we will not allow DCE to
// eliminate mutable ops.
//
// So, prefer to use the graph version if you can.
enum class DCESideEffectPolicy : uint8_t {
  // default behavior: dead code elimination will check if a node has side effects
  // and not delete it if it does.
  DONT_DELETE_NODES_WITH_SIDE_EFFECTS,
  // with this flag, dead code elimination will not check if a node has side
  // effects and treat nodes with side effects like any other node,
  // i.e. delete them if their outputs aren't used anywhere.
  ALLOW_DELETING_NODES_WITH_SIDE_EFFECTS
};

TORCH_API void EliminateDeadCode(const std::shared_ptr<Graph>& graph, DCESideEffectPolicy sideEffectPolicy = DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);
TORCH_API void EliminateDeadCode(Block* block, bool recurse = true, DCESideEffectPolicy sideEffectPolicy = DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);

// Invoke the user-provided callback on all live values before deleting anything
TORCH_API void EliminateDeadCode(
    Block* block,
    std::function<void(const std::unordered_set<const Value*>&)> cb,
    DCESideEffectPolicy sideEffectPolicy = DCESideEffectPolicy::DONT_DELETE_NODES_WITH_SIDE_EFFECTS);
} // namespace jit
} // namespace torch
