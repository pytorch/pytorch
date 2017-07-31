#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/init_pass.h"

#include <memory>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>


namespace torch { namespace jit { namespace tracer {

struct TracingState : public std::enable_shared_from_this<TracingState> {
  TracingState()
    : graph(new Graph()) {}

  std::unique_ptr<Graph> graph;
};

using torch::autograd::Variable;
using variable_list = std::vector<std::shared_ptr<Variable>>;

inline bool isTracing(const variable_list& vars) {
  for (auto& var : vars)
    if (!var->tracing_state.state.expired())
      return true;
  return false;
}

inline std::shared_ptr<TracingState> getTracingState(const variable_list& vars) {
  std::shared_ptr<TracingState> state;
  for (auto& var : vars) {
    auto var_state = var->tracing_state.state.lock();
    if (var_state) {
      if (!state) {
        state = var_state;
      } else if (state != var_state) {
        throw std::runtime_error("Mixing up traces");
      }
    }
  }
  JIT_ASSERT(state);
  return state;
}

inline void setValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, Node *node) {
  var->tracing_state.state = state;
  var->tracing_state.trace = node;
}

inline Node* getValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, bool mustExist = false) {
  auto var_state = var->tracing_state.state.lock();
  if (var_state) {
    JIT_ASSERT(var->tracing_state.state.lock() == state);
    return var->tracing_state.trace;
  }

  if (mustExist) throw std::runtime_error("untraced variable");

  Node *constant = state->graph->appendNewNode<Constant>(var->data);
  setValueTrace(state, var, constant);
  return constant;
}

inline std::shared_ptr<TracingState> enter(const variable_list& inputs) {
  auto state = std::make_shared<TracingState>();
  for (auto& input : inputs) {
    JIT_ASSERT(input->tracing_state.state.expired());
    input->tracing_state.state = state;
    input->tracing_state.trace = state->graph->addInput();
  }
  // TODO: register exit hooks!
  return state;
}

inline void exit(const std::shared_ptr<TracingState>& state, const variable_list& outputs) {
  for (auto& output : outputs) {
    JIT_ASSERT(output->tracing_state.state.lock() == state);
    state->graph->registerOutput(getValueTrace(state, output, true));
    output->tracing_state.state.reset();
    output->tracing_state.trace = nullptr;
  }
  // TODO: register enter hooks!
}

}}} // namespace torch::jit::tracer
