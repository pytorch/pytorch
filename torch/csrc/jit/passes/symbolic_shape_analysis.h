#pragma once

#include <c10/util/variant.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/ir/ir.h>
#include <unordered_map>

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
      : partial_eval_shape_graph(partial_eval_shape_graph),
        enclosing_graph_value_to_shape_graph_input_(
            enclosing_graph_value_to_shape_graph_input),
        graph_output_to_symbolic_shape_dim_(
            graph_output_to_symbolic_shape_dim){};

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
TORCH_API std::unique_ptr<std::vector<c10::SymbolicShape>>
calculateSymbolicShapesOnOp(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& inputs);

struct TORCH_API CanonicalizedSymbolicShape {
  CanonicalizedSymbolicShape(c10::SymbolicShape orig_shape) {
    auto sizes = orig_shape.sizes();
    if (!sizes) {
      values_ = c10::nullopt;
      return;
    }
    values_ = std::vector<int64_t>();
    int64_t cur_symbolic_index = -1;
    for (int64_t cur_loc = 0; cur_loc < sizes->size(); cur_loc++) {
      auto& cur_shape = sizes->at(cur_loc);
      if (cur_shape.is_static()) {
        is_symbolic_.emplace_back(false);
        values_->push_back(cur_shape.static_size());
      } else {
        // Check for aliasing
        is_symbolic_.emplace_back(true);
        int64_t to_check = 0;
        for (; to_check < cur_loc; to_check++) {
          auto& shape_to_check = sizes->at(cur_loc);
          if (shape_to_check == cur_shape) {
            values_->push_back(values_->at(to_check));
            break;
          }
        }
        if (to_check == cur_loc) {
          // Didin't find an aliasing ss
          values_->push_back(cur_symbolic_index);
          cur_symbolic_index--;
        }
      }
    }
  }

 private:
  c10::optional<std::vector<int64_t>> values_;
  std::vector<bool> is_symbolic_;

  friend bool operator==(
      const CanonicalizedSymbolicShape& a,
      const CanonicalizedSymbolicShape b) {
    if (a.values_.has_value() != b.values_.has_value()) {
      return false;
    }
    if (!a.values_.has_value()) {
      return true;
    }
    return (
        a.values_.value() == b.values_.value() &&
        a.is_symbolic_ == b.is_symbolic_);
  };
};
} // namespace jit
} // namespace torch
