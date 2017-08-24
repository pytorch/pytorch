#pragma once

#include <Python.h>
#include <memory>
#include <string>
#include <mutex>

#include "torch/csrc/autograd/function.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/engine.h"

namespace torch { namespace autograd {

struct EvalOutput : Function {
  EvalOutput(const edge_type& next_edge)
    : next_edge(next_edge) {
    num_inputs = 1;
    is_executable = next_edge.first->is_executable;
  }

  virtual variable_list apply(const variable_list& inputs) override {
    throw std::logic_error("EvalOutput::apply() called");
  }

  edge_type next_edge;
};

struct Eval : Function {
  using edge_set = std::unordered_set<edge_type, edge_hasher>;
  using edge_order = std::unordered_map<edge_type, int, edge_hasher>;
  using placeholder_list = std::vector<std::shared_ptr<EvalOutput>>;

  // This struct has only one member, but it's useful to e.g. add a set of all
  // nodes when debugging this stuff, so I'm leaving it as is.
  struct Subgraph {
    struct Boundary {
      // All nodes from within the subgraph that connect to the outside.
      // These are the places that will need to be patched to point to placeholders.
      // Contains pairs of (fn, offset into next_functions).
      edge_set begins;
      // All nodes that are not in the subgraph, but are in the union of
      // next_functions of all nodes from the subgraph. These are the places that
      // will be modeled by placeholders.
      // Contains pairs of (fn, input_nr) and is equivalent to next_functions
      // of an Eval node that will replace the subgraph.
      edge_set ends;
    };

    Boundary boundary;
  };

  virtual inline bool is_traceable() final { return traceable; }

  virtual variable_list apply(const variable_list& inputs) override;

  void replaceSubgraph(
      const variable_list& inputs,
      const variable_list& outputs,
      const placeholder_list& inherited_placeholders = placeholder_list());

  static variable_list filterRelevantOutputs(const variable_list& inputs, const variable_list& outputs);
  edge_order computeInputOrder(const variable_list& inputs, const placeholder_list& inherited_placeholders);

  static std::shared_ptr<Eval> getBackwardEval(const variable_list& inputs, const variable_list& outputs) {
    auto relevant_outputs = filterRelevantOutputs(inputs, outputs);
    if (relevant_outputs.size() == 0)
      return nullptr;
    return std::dynamic_pointer_cast<Eval>(relevant_outputs[0]->grad_fn);
  }

  function_list roots;
  placeholder_list placeholders;
  jit::Node* forward_ctx_select = nullptr;
  bool traceable = false;

private:
  Engine::pre_callback_map getCallbacks(variable_list& outputs, std::mutex& outputs_mutex);

  Subgraph getSubgraph(
      const variable_list& inputs,
      const variable_list& outputs,
      const placeholder_list& inherited_placeholders);
};

}} // namespace torch::autograd
