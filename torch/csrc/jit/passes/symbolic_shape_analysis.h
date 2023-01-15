#pragma once

#include <c10/util/variant.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {

// CAUTION NOT TO BE USED, STILL A WIP, NOT STABLE

TORCH_API void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph);

// CAUTION NOT TO BE USED, STILL A WIP, NOT STABLE
// From [beg, end) attempt to propagate shapes and
// build up a graph that will compute all remaining symbolic
// shapes in [beg, end) that can be executed before beg

struct ShapeComputeGraphMapping {
  ShapeComputeGraphMapping(
      std::shared_ptr<Graph> partial_eval_shape_graph,
      std::unordered_map<Value*, Value*>
          enclosing_graph_value_to_shape_graph_input,
      std::unordered_map<Value*, int64_t> graph_output_to_symbolic_shape_dim)
      : partial_eval_shape_graph(std::move(partial_eval_shape_graph)),
        enclosing_graph_value_to_shape_graph_input_(
            std::move(enclosing_graph_value_to_shape_graph_input)),
        graph_output_to_symbolic_shape_dim_(
            std::move(graph_output_to_symbolic_shape_dim)){};

  std::shared_ptr<Graph> partial_eval_shape_graph;
  std::unordered_map<Value*, Value*>
      enclosing_graph_value_to_shape_graph_input_;
  std::unordered_map<Value*, int64_t> graph_output_to_symbolic_shape_dim_;
};

TORCH_API c10::optional<ShapeComputeGraphMapping>
PropagateShapesAndBuildLargeShapeComputeGraph(
    std::shared_ptr<Graph>& graph,
    Node* beg,
    Node* end);

// don't insert complete tensor shapes in shape compute graphs and instead
// rely on our partial evaluation pipeline to propagate information.
// this is a good proxy for our ability to propagate non-complete shape
// information.
TORCH_API bool setSymbolicShapeAnalysisTestMode(bool value);
TORCH_API bool symbolicShapeAnalysisTestModeEnabled();

using SSAInput = c10::variant<IValue, c10::SymbolicShape>;
TORCH_API c10::optional<std::vector<c10::SymbolicShape>>
calculateSymbolicShapesOnOp(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& inputs);
} // namespace jit
} // namespace torch
