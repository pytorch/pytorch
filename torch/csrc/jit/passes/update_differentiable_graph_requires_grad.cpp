#include <torch/csrc/jit/passes/update_differentiable_graph_requires_grad.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

namespace torch {
namespace jit {

void UpdateDifferentiableGraphRequiresGrad(
    Block* block,
    c10::optional<bool> new_requires_grad) {
  for (Node* n : block->nodes()) {
    for (Value* v : n->inputs()) {
      auto ty = v->type()->cast<TensorType>();
      if (ty) {
        v->setType(ty->withRequiresGrad(new_requires_grad));
      }
    }
    if (n->kind() == prim::profile) {
      n->ty_(
          attr::profiled_type,
          n->ty(attr::profiled_type)
              ->expectRef<TensorType>()
              .withRequiresGrad(new_requires_grad));
    }
    for (Block* b : n->blocks()) {
      UpdateDifferentiableGraphRequiresGrad(b, new_requires_grad);
    }
  }
}

void UpdateDifferentiableGraphRequiresGrad(
    std::shared_ptr<Graph>& diff_forward_graph,
    c10::optional<bool> new_requires_grad) {
  UpdateDifferentiableGraphRequiresGrad(
      diff_forward_graph->block(), new_requires_grad);
}

} // namespace jit
} // namespace torch
