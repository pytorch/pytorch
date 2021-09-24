#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>

namespace torch {
namespace jit {
namespace {

using Tensor = at::Tensor;

class ConcatLinearLayers {
  std::shared_ptr<Graph> graph_;
  bool graph_modified = false;

  bool is_constant_linear_op(Node* node) {
    if (node->kind() != aten::linear) {
      return false;
    }
    auto weight = node->namedInput("weight");
    if (weight->type() == NoneType::get()) {
      return false;
    }
    
    // This should also filter out out-variants of the linear op.
    if (nonConstantParameters(node)) {
      return false;
    }
    auto weight_tensor = constant_as<Tensor>(weight).value();
    // Op is only profitable on CUDA
    return weight_tensor.device().is_cuda();
  }

  void replace_linear_with_matmul(Node* node) {
    graph_modified = true;
    Node* matmul = nullptr;

    {
      WithInsertPoint insert_guard(node);
      auto weight = node->namedInput("weight");
      Tensor weight_tensor = constant_as<Tensor>(weight).value();
      Tensor weight_t_tensor = weight_tensor.transpose(0, 1);

      Value* weight_t = graph_->insertConstant(weight_t_tensor);
      matmul = graph_->create(aten::matmul, {node->inputs()[0], weight_t});
      matmul->insertAfter(node);
    }

    // Handle a bias if there is any
    auto bias = node->namedInput("bias");
    if (bias->type() == NoneType::get()) {
      node->replaceAllUsesWith(matmul);
    } else {
      Node* bias_result = graph_->create(aten::add, {matmul->output(), bias});
      bias_result->insertAfter(matmul);
      node->replaceAllUsesWith(bias_result);
    }
    node->destroy();
  };

public:
  bool handleBlockAndSubblocks(Block* block) {
    for (auto node : block->nodes()) {
      for (Block* block : node->blocks()) {
        handleBlockAndSubblocks(block);
      }

      if (is_constant_linear_op(node)) {
        replace_linear_with_matmul(node);
      }
    }
    return graph_modified;
  }

};
} // namespace

TORCH_API bool FrozenLinearTranspose(std::shared_ptr<Graph>& graph) {
  ConcatLinearLayers concatLayers;
  GRAPH_DUMP("Before FrozenLinearTranspose", graph);
  bool changed = concatLayers.handleBlockAndSubblocks(graph->block());
  if (changed) {
    GRAPH_DUMP("After FrozenLinearTranspose", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch
