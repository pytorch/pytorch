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
    : graph(new Graph())
    , active(false) {}

  std::unique_ptr<Graph> graph;
  bool active;
};

using torch::autograd::Variable;
using variable_list = std::vector<std::shared_ptr<Variable>>;

inline bool isTracing(const variable_list& vars) {
  for (auto& var : vars) {
    if (!var) continue;
    auto state = var->tracing_state.state.lock();
    if (state && state->active)
        return true;
  }
  return false;
}

inline std::shared_ptr<TracingState> getTracingState(const variable_list& vars) {
  std::shared_ptr<TracingState> state;
  for (auto& var : vars) {
    if (!var) continue;
    auto var_state = var->tracing_state.state.lock();
    if (var_state) {
      if (!state) {
        state = var_state;
      }
      JIT_ASSERT(state == var_state);
    }
  }
  JIT_ASSERT(state);
  return state;
}

// TODO: what if an output is used in an in-place op? it might appear in the trace again,
// but it really points to a different place in the graph than its trace
inline void setValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, Node *node) {
  JIT_ASSERT(var->tracing_state.state.lock() == state || var->tracing_state.state.expired());
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
  state->active = true;
  // TODO: register exit hooks!
  return state;
}

inline void exit(const variable_list& outputs) {
  auto state = getTracingState(outputs);
  for (auto& output : outputs) {
    state->graph->registerOutput(getValueTrace(state, output, true));
  }
  state->active = false;
  // TODO: register enter hooks!
}

}}} // namespace torch::jit::tracer
