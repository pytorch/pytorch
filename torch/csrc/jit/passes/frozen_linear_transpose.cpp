#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <iostream>

namespace torch {
namespace jit {
namespace {

using Tensor = at::Tensor;

class TransposeFrozenLinear {
 public:
  TransposeFrozenLinear(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run() {
    handleBlockAndSubblocks(graph_->block());
    return graph_modified;
  }

  bool is_constant_linear_op(Node* node) {
    if (node->kind() != aten::linear) {
      return false;
    }
    auto weight = node->namedInput("weight");
    if (weight->type() == NoneType::get()) {
      return false;
    }

    // This also filters out out-variants of the linear op.
    if (nonConstantParameters(node)) {
      return false;
    }

    // Op is only profitable on CUDA
    auto weight_tensor = constant_as<Tensor>(weight).value();
    return weight_tensor.device().is_cuda();
  }

  void replace_linear_with_matmul(Node* node) {
    graph_modified = true;
    Node* matmul = nullptr;

    {
      WithInsertPoint insert_guard(node);
      auto weight = node->namedInput("weight");

      Tensor weight_tensor = constant_as<Tensor>(weight).value();
      Tensor weight_t_tensor = at::transpose(weight_tensor, 1, 0)
                                   .clone(at::MemoryFormat::Contiguous);
      Value* weight_t = graph_->insertConstant(weight_t_tensor);
      matmul = graph_->create(aten::matmul, {node->inputs()[0], weight_t});
      matmul->insertAfter(node);
    }

    // Handle a bias if there is any
    WithInsertPoint insert_guard(matmul);
    auto bias = node->namedInput("bias");
    if (bias->type() == NoneType::get()) {
      node->replaceAllUsesWith(matmul);
    } else {
      Value* bias_scale = graph_->insertConstant(1);
      Node* bias_result =
          graph_->create(aten::add, {matmul->output(), bias, bias_scale});
      bias_result->insertAfter(matmul);
      node->replaceAllUsesWith(bias_result);
    }
    node->destroy();
  };

  void handleBlockAndSubblocks(Block* block) {
    // Can't delete nodes while also iterating over it
    std::vector<Node*> constant_linear_nodes;

    for (auto node : block->nodes()) {
      for (Block* block : node->blocks()) {
        handleBlockAndSubblocks(block);
      }

      if (is_constant_linear_op(node)) {
        constant_linear_nodes.push_back(node);
      }
    }
    for (auto node : constant_linear_nodes) {
      replace_linear_with_matmul(node);
    }
    return;
  }

 private:
  std::shared_ptr<Graph> graph_;
  bool graph_modified = false;
};
} // namespace

TORCH_API bool FrozenLinearTranspose(std::shared_ptr<Graph>& graph) {
  TransposeFrozenLinear transposeWeight(graph);
  GRAPH_DUMP("Before FrozenLinearTranspose", graph);
  bool changed = transposeWeight.run();
  if (changed) {
    GRAPH_DUMP("After FrozenLinearTranspose", graph);
  }
  return changed;
}

} // namespace jit
} // namespace torch
