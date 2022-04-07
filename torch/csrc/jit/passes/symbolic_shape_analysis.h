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
TORCH_API c10::optional<std::vector<c10::SymbolicShape>>
calculateSymbolicShapesOnOp(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& inputs);

struct TORCH_API CanonicalizedSymbolicShape {
  CanonicalizedSymbolicShape(
      const c10::SymbolicShape& orig_shape,
      std::unordered_map<int64_t, int64_t>& ss_map) {
    init(orig_shape, ss_map);
  }

  CanonicalizedSymbolicShape(c10::SymbolicShape& orig_shape) {
    std::unordered_map<int64_t, int64_t> new_ssmap;
    init(orig_shape, new_ssmap);
  }

  size_t hash() const {
    if (!values_.has_value()) {
      return 98024139491283;
    }
    return c10::hash<std::vector<int64_t>>()(values_.value());
  }

  c10::SymbolicShape toSymbolicShape(
      std::unordered_map<int64_t, int64_t> inverse_ss_map) const;

 private:
  c10::optional<std::vector<int64_t>> values_;

  void init(
      const c10::SymbolicShape& orig_shape,
      std::unordered_map<int64_t, int64_t>& ss_map) {
    auto sizes = orig_shape.sizes();
    if (!sizes) {
      values_ = c10::nullopt;
      return;
    }
    values_ = std::vector<int64_t>();
    int64_t cur_symbolic_index = -(int64_t)ss_map.size() - 1;
    for (auto& cur_shape : *sizes) {
      if (cur_shape.is_static()) {
        values_->push_back(cur_shape.static_size());
      } else {
        // Check for aliasing
        auto it = ss_map.find(cur_shape.value());

        if (it == ss_map.end()) {
          values_->push_back(cur_symbolic_index);
          ss_map.insert({cur_shape.value(), cur_symbolic_index});
          cur_symbolic_index--;
        } else {
          values_->push_back(it->second);
        }
      }
    }
  }

  friend bool operator==(
      const CanonicalizedSymbolicShape& a,
      const CanonicalizedSymbolicShape& b) {
    if (a.values_.has_value() != b.values_.has_value()) {
      return false;
    }
    if (!a.values_.has_value()) {
      return true;
    }
    return a.values_.value() == b.values_.value();
  };
};

TORCH_API void clear_shape_cache();
TORCH_API size_t get_shape_cache_size();

} // namespace jit
} // namespace torch
