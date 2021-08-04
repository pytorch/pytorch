#include <torch/csrc/jit/passes/eliminate_tuple_construct_unpack.h>

namespace torch {
namespace jit {

void EliminateTupleConstructUnpack(std::shared_ptr<torch::jit::Graph>& graph) {
  auto block = graph->block();
  for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); ++it) {
    auto* node = *it;
    if (node->kind() != prim::TupleUnpack) {
      continue;
    }
    auto* src = node->input()->node();
    if (src->kind() != prim::TupleConstruct) {
      continue;
    }

    TORCH_CHECK(
        src->inputs().size() == node->outputs().size(),
        "The number of TupleConstruct inputs and the number of TupleUnpack outputs should be same.");
    size_t nInputs = src->inputs().size();
    for (size_t i = 0; i < nInputs; ++i) {
      node->output(i)->replaceAllUsesWith(src->input(i));
    }
  }
}

void EliminateTupleUnpackConstruct(std::shared_ptr<torch::jit::Graph>& graph) {
  auto block = graph->block();
  for (auto it = block->nodes().rbegin(); it != block->nodes().rend(); ++it) {
    auto* node = *it;
    if (node->kind() != prim::TupleConstruct) {
      continue;
    }
    size_t nInputs = node->inputs().size();
    auto* src0 = node->input(0)->node();
    if (src0->outputs().size() != nInputs ||
        src0->kind() != prim::TupleUnpack) {
      continue;
    }
    bool shouldEliminate = true;
    for (size_t i = 0; i < nInputs; ++i) {
      auto* src = node->input(i)->node();
      if (src != src0 || node->input(i) != src->output(i)) {
        shouldEliminate = false;
        break;
      }
    }
    if (shouldEliminate) {
      node->output()->replaceAllUsesWith(src0->input());
    }
  }
}

} // namespace jit
} // namespace torch
