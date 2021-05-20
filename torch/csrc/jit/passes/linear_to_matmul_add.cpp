#include <algorithm>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/linear_to_matmul_add.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>

namespace torch {
namespace jit {

namespace {

// Actually modify the IR; called after making the appropriate checks
void PerformLinearNodeDecomposition(Node* node) {
  auto g = node->owningGraph();
  auto t_weight_n =
      g->create(aten::t, {node->namedInput("weight")}, 1)->insertBefore(node);
  auto matmul_n =
      g->create(
           aten::matmul, {node->namedInput("input"), t_weight_n->output()}, 1)
          ->insertBefore(node);
  auto add_n =
      g->create(aten::add, {matmul_n->output(), node->namedInput("bias")}, 1)
          ->insertBefore(node);
  node->output()->replaceAllUsesWith(add_n->output());
  node->destroy();
}

bool LinearIsGPU(const Node* linear_node) {
  auto device =
      linear_node->namedInput("weight")->type()->cast<TensorType>()->device();
  return device && device->is_cuda();
}

// Recurse through each block and perform the linear decomposition.
void DecomposeLinearToMatmulAdd(Block* b, bool restrict_to_gpu) {
  std::vector<Node*> linear_nodes;
  for (Node* n : b->nodes()) {
    for (auto* child_block : n->blocks()) {
      DecomposeLinearToMatmulAdd(child_block, restrict_to_gpu);
    }
    if (n->kind() == aten::linear) {
      linear_nodes.push_back(n);
    }
  }

  for (Node* node : linear_nodes) {
    // If the weights are GPU (unless restrict_to_gpu == false),
    // and if all its uses are supported for fusion,
    // and if bias is present (so a non-trivial add node would be created)
    // then decompose into matmul + add
    if (!restrict_to_gpu || LinearIsGPU(node)) {
      auto uses = node->output()->uses();
      if (std::all_of(uses.begin(), uses.end(), [](const Use& u) {
            return tensorexpr::isSupported(u.user);
          })) {
        if (!node->namedInput("bias")->mustBeNone()) {
          PerformLinearNodeDecomposition(node);
        }
      }
    }
  }
}

} // namespace

// Replace Linear nodes with Matmul + Add whenever that would enable
// Add/* fusion to occur.
// For testing purposes, a boolean argument allows us to do this for nodes
// that would be executed on either GPU or CPU.
// In production, we only want this enabled for GPU;
// accordingly, header gives default value restrict_to_gpu = true
void DecomposeLinearToMatmulAdd(
    std::shared_ptr<Graph>& graph,
    bool restrict_to_gpu) {
  DecomposeLinearToMatmulAdd(graph->block(), restrict_to_gpu);
}

} // namespace jit
} // namespace torch
