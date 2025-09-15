#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_linear_transpose.h>
#include <torch/csrc/jit/passes/utils/optimization_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/transpose.h>
#endif

#include <iostream>
#include <utility>

namespace torch::jit {
namespace {

using Tensor = at::Tensor;

class TransposeFrozenLinear {
 public:
  TransposeFrozenLinear(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  bool run() {
    // Can't delete nodes while also iterating over it
    DepthFirstGraphNodeIterator graph_it(graph_);

    for (auto next_node = graph_it.next(); next_node != nullptr;) {
      Node* node = next_node;
      next_node = graph_it.next();

      if (is_constant_linear_op(node)) {
        replace_linear_with_matmul(node);
      }
    }
    return graph_modified_;
  }

  bool is_constant_linear_op(Node* node) {
    if (node->kind() != aten::linear) {
      return false;
    }

    // This also filters out out-variants of the linear op.
    return !nonConstantParameters(node);
  }

  void replace_linear_with_matmul(Node* node) {
    graph_modified_ = true;
    Node* matmul = nullptr;

    {
      WithInsertPoint insert_guard(node);
      auto weight = node->namedInput("weight");

      Tensor weight_tensor = constant_as<Tensor>(weight).value();
      Tensor weight_t_tensor = at::transpose(weight_tensor, 1, 0)
                                   .clone(at::MemoryFormat::Contiguous);
      Value* weight_t = graph_->insertConstant(std::move(weight_t_tensor));
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
  }

  void handleBlockAndSubblocks(Block* block) {}

 private:
  std::shared_ptr<Graph> graph_;
  bool graph_modified_ = false;
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

} // namespace torch::jit
