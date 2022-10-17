// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <caffe2/torch/csrc/jit/backends/xnnpack/xnnpack_graph_builder.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <xnnpack.h>

// graph passes
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/runtime/jit_trace.h>
#include <torch/csrc/jit/tensorexpr/graph_opt.h>

namespace torch {
namespace jit {
namespace xnnpack {
namespace delegate {

std::shared_ptr<torch::jit::Graph> XNNGraph::optimizeAndTraceGraph(
    std::shared_ptr<torch::jit::Graph> graph,
    std::vector<c10::IValue>& example_inputs) {
  graph = tensorexpr::removeUnusedSelfArgument(graph);
  OptimizeFrozenGraph(graph, true);
  RemoveListMutation(graph);
  RemoveTensorMutation(graph);
  LowerAllTuples(graph);
  ConstantPropagation(graph);
  graph = TraceGraph(graph, example_inputs);

  return graph;
}

void XNNGraph::buildXNNGraph(
    std::shared_ptr<torch::jit::Graph>& graph,
    std::vector<c10::IValue> example_inputs) {
  graph = optimizeAndTraceGraph(graph, example_inputs);
  checkOpsToDelegate(graph);
}

void XNNGraph::checkOpsToDelegate(std::shared_ptr<torch::jit::Graph>& graph) {
  std::unordered_set<string> unsupported_ops;
  DepthFirstGraphNodeIterator it(graph);
  Node* node = nullptr;
  while ((node = it.next()) != nullptr) {
    switch (node->kind()) {
      case prim::Constant:
      case aten::add: {
        break;
      }
      default: {
        unsupported_ops.insert(node->kind().toDisplayString());
      }
    }
  }
  std::stringstream error;
  for (auto itr = unsupported_ops.begin(); itr != unsupported_ops.end();
       itr++) {
    error << *itr << std::endl;
    ;
  }
  TORCH_CHECK(
      unsupported_ops.empty(),
      "the module contains the following unsupported ops:\n" + error.str());
}

} // namespace delegate
} // namespace xnnpack
} // namespace jit
} // namespace torch
