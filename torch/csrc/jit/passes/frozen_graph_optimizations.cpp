#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/utils/memory.h>

namespace torch {
namespace jit {

void OptimizeFrozenGraph(std::shared_ptr<Graph>& graph) {
  FoldFrozenConvBatchnorm(graph);
  FoldFrozenConvAddOrSub(graph);
  FoldFrozenConvMulOrDiv(graph);
}

} // namespace jit
} // namespace torch
