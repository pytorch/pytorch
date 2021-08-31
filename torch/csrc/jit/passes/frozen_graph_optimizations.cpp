#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>
#include <vector>
#include "ATen/core/functional.h"
#include "ATen/core/interned_strings.h"

namespace torch {
namespace jit {

bool nonConstantParameters(Node* n) {
  // Checks if the parameters, not including the
  // first param are all constants.
  for (size_t i = 1; i < n->inputs().size(); i++) {
    if (n->inputs().at(i)->node()->kind() != prim::Constant) {
      return true;
    }
  }
  return false;
}

void OptimizeFrozenGraph(
    std::shared_ptr<Graph>& graph,
    bool optimize_numerics) {
  removeDropout(graph);
  // run a couple times to capture Conv -> Mul -> Add etc
  if (optimize_numerics) {
    for (const auto i : c10::irange(2)) {
      (void)i; // Suppress unused variable warning
      FoldFrozenConvBatchnorm(graph);
      FoldFrozenConvAddOrSub(graph);
      FoldFrozenConvMulOrDiv(graph);
    }
    FrozenConcatLinear(graph);
  }
}

} // namespace jit
} // namespace torch
