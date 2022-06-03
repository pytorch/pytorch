#include <gtest/gtest.h>

#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"
#include "torch/csrc/jit/runtime/graph_iterator.h"

namespace torch {
namespace jit {

TEST(CreateAutodiffSubgraphsTest, Basic) {
  auto graph = build_lstm();
  auto add_requires_grad_to_value = [](Value* v) {
    if (v->type()->kind() == TypeKind::TensorType) {
      v->setType(v->type()->expectRef<TensorType>().withRequiresGrad(true));
    }
  };

  DepthFirstGraphNodeIterator it(graph);
  Node* n = nullptr;
  while ((n = it.next()) != nullptr) {
    for (Value* v : n->outputs()) {
      add_requires_grad_to_value(v);
    }
  }
  for (Value* v : graph->inputs()) {
    add_requires_grad_to_value(v);
  }

  CreateAutodiffSubgraphs(graph, /*threshold=*/2);
  // all of the ops are within the DifferentiableGraph
  testing::FileCheck()
      .check_not("aten::mm")
      ->check_not("aten::sigmoid")
      ->check_not("aten::tanh")
      ->check_not("aten::mul")
      ->check("DifferentiableGraph")
      ->check_next("return")
      ->run(*graph);
}

} // namespace jit
} // namespace torch
