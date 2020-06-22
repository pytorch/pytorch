#include <torch/csrc/jit/passes/verify_shape_analysis.h>
#include <ATen/core/interned_strings.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

namespace torch {
namespace jit {

RegisterOperators reg({Operator(
    "aten::check_tensor(Tensor(a!) x) -> None",
    [](Stack& s) { return 0; },
    aliasAnalysisFromSchema())});

// NOT FOR LANDING YET THUS COPY PASTA
void VerifyShapeAnalysis(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end(); it++) {
    Node * n = *it;
    for (Block* b : n->blocks()) {
      VerifyShapeAnalysis(b);
    }
    if (n->kind() == aten::check_tensor) {
      continue;
    }
    {
      WithInsertPoint guard(n);
      std::unordered_set<Value*> tensor_inputs;
      for (auto input : n->inputs()) {
        if (input->type()->cast<TensorType>()) {
          tensor_inputs.insert(input);
        }
      }
      for (Value* v : tensor_inputs) {
        auto node = n->owningGraph()->create(aten::check_tensor, {v}, 1);
        node->insertBefore(n);
      }
    }
    {
      WithInsertPoint guard(n);
      std::unordered_set<Value*> tensor_outputs;
      for (auto output : n->outputs()) {
        if (output->type()->cast<TensorType>()) {
          tensor_outputs.insert(output);
        }
      }
      for (Value* v : tensor_outputs) {
        auto node = n->owningGraph()->create(aten::check_tensor, {v}, 1);
        node->insertAfter(n);
      }
    }
  }
}

void VerifyShapeAnalysis(const std::shared_ptr<Graph>& graph) {
  VerifyShapeAnalysis(graph->block());
}

} // namespace jit
} // namespace torch
