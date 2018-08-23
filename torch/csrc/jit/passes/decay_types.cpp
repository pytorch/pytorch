#include "torch/csrc/jit/passes/decay_types.h"

namespace torch { namespace jit {

void DecayTypes(Block * b) {
  for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
    Node * n = *it;

    for (Block * b : n->blocks()) {
      DecayTypes(b);
    }

    for (Value *out : n->outputs()) {
      if (out->type()->cast<TensorType>()) {
        out->setType(DynamicType::get());
      }
    }
  }
}

void DecayTypes(const std::shared_ptr<Graph>& graph) {
  for (Value * input : graph->inputs()) {
    if (input->type()->cast<TensorType>()) {
      input->setType(DynamicType::get());
    }
  }
  DecayTypes(graph->block());
}


}}  // namespace torch::jit
