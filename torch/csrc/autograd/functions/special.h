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
    // It would be nice if we could inherit this from the function of next_edge,
    // but we want to always run this node to capture the output. This might
    // confuse some of the functions causing them to do unnecessary work.
    // TODO: it should be possible to improve this once we get rid of NULL Variables
    is_executable = true;
    if (next_edge.first) {
      input_sizes.emplace_back(next_edge.first->input_sizes.at(next_edge.second));
    } else {
      input_sizes.emplace_back(nullptr);
    }
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

  // Roots are empty if simple_graph is not NULL.
  // simple_graph is an optimization of first backward stage - in this case
  // all Eval subgraphs contain only a single gradient function, and the
  // graph search on creation + call to the engine in apply can be elided
  function_list roots;
  std::shared_ptr<Function> simple_graph;

  placeholder_list placeholders;
  jit::Node* forward_ctx_select = nullptr;
  bool traceable = false;

private:
  std::pair<function_list, variable_list> filterRoots(const variable_list& inputs);
  Engine::pre_callback_map getCallbacks(variable_list& outputs, std::mutex& outputs_mutex);

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
