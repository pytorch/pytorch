#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"

namespace torch {
namespace jit {

void testCreateAutodiffSubgraphs() {
  auto graph = build_lstm();
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
