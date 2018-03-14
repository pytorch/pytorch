#pragma once

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/engine.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace torch { namespace autograd {

struct EvalOutput : Function {
  explicit EvalOutput(const Edge& next_edge_)
      : Function(/*num_inputs=*/1), next_edge(next_edge_) {}

  virtual variable_list apply(const variable_list& inputs) override {
    throw std::logic_error("EvalOutput::apply() called");
  }

  Edge next_edge;
};

struct Eval : Function {
  using edge_set = std::unordered_set<Edge>;
  using edge_order = std::unordered_map<Edge, int>;
  using placeholder_list = std::vector<std::shared_ptr<EvalOutput>>;

  // This struct has only one member, but it's useful to e.g. add a set of all
  // nodes when debugging this stuff, so I'm leaving it as is.
  struct Subgraph {
    struct Boundary {
      // All nodes from within the subgraph that connect to the outside.
      // These are the places that will need to be patched to point to placeholders.
      // Contains pairs of (fn, offset into next_edges).
      edge_set begins;
      // All nodes that are not in the subgraph, but are in the union of
      // next_edges of all nodes from the subgraph. These are the places that
      // will be modeled by placeholders.
      // Contains pairs of (fn, input_nr) and is equivalent to next_edges
      // of an Eval node that will replace the subgraph.
      edge_set ends;
    };

    Boundary boundary;
  };

  virtual ~Eval() {}

  virtual inline bool is_traceable() final { return traceable; }

  virtual variable_list apply(const variable_list& inputs) override;

  bool replaceSubgraph(
      const variable_list& inputs,
      const variable_list& outputs,
      const placeholder_list& inherited_placeholders = placeholder_list());

  static variable_list filterRelevantOutputs(const variable_list& inputs, const variable_list& outputs);
  edge_order computeInputOrder(const variable_list& inputs, const placeholder_list& inherited_placeholders);

  static std::shared_ptr<Eval> getBackwardEval(const variable_list& inputs, const variable_list& outputs) {
    auto relevant_outputs = filterRelevantOutputs(inputs, outputs);
    if (relevant_outputs.size() == 0)
      return nullptr;
    return std::dynamic_pointer_cast<Eval>(relevant_outputs[0].grad_fn());
  }

  virtual std::shared_ptr<Eval> newEval() {
    return std::make_shared<Eval>();
  }

  // Roots are empty if simple_graph is not nullptr.
  // simple_graph is an optimization of first backward stage - in this case
  // all Eval subgraphs contain only a single gradient function, and the
  // graph search on creation + call to the engine in apply can be elided
  edge_list roots;
  std::shared_ptr<Function> simple_graph;

  placeholder_list placeholders;
  jit::Value* forward_ctx_select = nullptr;
  bool traceable = false;

private:
  std::pair<edge_list, variable_list> filterRoots(const variable_list& inputs);

  Subgraph getSubgraph(
      const variable_list& inputs,
      const variable_list& outputs,
      const placeholder_list& inherited_placeholders);

  bool trySimpleEval(
      const variable_list& inputs,
      const variable_list& outputs,
      const placeholder_list& inherited_placeholders);
};

}} // namespace torch::autograd
