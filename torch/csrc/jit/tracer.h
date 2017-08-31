#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/jit/assert.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/init_pass.h"

#include <memory>
#include <mutex>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>


namespace torch { namespace jit { namespace tracer {

using torch::autograd::Variable;
using variable_list = std::vector<std::shared_ptr<Variable>>;

namespace detail {

inline ValueTracingStateElem* getValueState(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, bool alloc = true) {
  for (auto it = var->tracing_state->begin(); it != var->tracing_state->end();) {
    auto ts = it->state.lock();
    // GC of invalidated tracing states
    if (!ts) {
      auto current_it = it++;
      var->tracing_state->erase(current_it);
      continue;
    } else if (ts == state) {
      return &(*it);
    }
    ++it;
  }
  if (alloc) {
    var->tracing_state->emplace_front();
    auto & vts = var->tracing_state->front();
    vts.state = state;
    return &vts;
  } else {
    return nullptr;
  }
}

inline bool isElemActive(const ValueTracingStateElem& vts) {
  auto state = vts.state.lock();
  return state && state->active;
}

}


// Should a function which takes 'vars' as inputs be traced?
// It sufficies for ONE variable to be tracing: any "untraced" variables
// are treated as constants.
inline bool isTracing(const variable_list& vars) {
  for (auto& var : vars) {
    if (!var || !var->tracing_state) continue;
    if (std::any_of(var->tracing_state->begin(), var->tracing_state->end(), detail::isElemActive))
      return true;
  }
  return false;
}

// Retrieve the tracing state which a function applied with 'vars' should
// be recorded to.  Precondition: isTracing(vars) == true.  At the moment,
// we don't support mixing up variables from different traces; this code
// will need to be revisited if that ever becomes supported.
inline std::shared_ptr<TracingState> getTracingState(const variable_list& vars) {
  std::shared_ptr<TracingState> state;
  for (auto& var : vars) {
    if (!var || !var->tracing_state) continue;
    for (auto & vts : *var->tracing_state) {
      auto var_state = vts.state.lock();
      if (!var_state || !var_state->active) continue;
      if (!state) state = var_state;
      JIT_ASSERT(var_state == state);
    }
  }
  JIT_ASSERT(state);
  return state;
}

// Having finished adding a new 'node' to the graph IR owned by TracingState 'state',
// 'setValueTrace' associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
inline void setValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, Node *node) {
  auto vts = detail::getValueState(state, var);
  vts->trace = node;
}

// Given a variable 'var', return the 'node' which represents the instruction
// which computes the value of this variable in the IR.  When 'mustExist' is
// false, we interpret untraced variables as constants that are just embedded
// in the graph.  This is useful to handle code which does things like this
// (from torch.autograd.variable):
//
//    def mm(self, matrix):
//      output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
//      return Addmm.apply(output, self, matrix, 0, 1, True)
//
// Here, mm fakes up a dummy variable with uninitialized data to do an inplace
// update on, but subsequently ignores it because the alpha scaling factor is zero.
// This is one of the cases where a Variable can be created inside of a trace, and
// if we treat it as a constant, everything will work out.
inline Node* getValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, bool mustExist = false) {
  if (!var) {
    return state->graph->appendNode(state->graph->createUndefined());
  }
  if (mustExist) {
    auto vts = detail::getValueState(state, var, false);
    if (!vts) throw std::runtime_error("untraced variable");
    return vts->trace;
  }

  auto vts = detail::getValueState(state, var, true);
  if (vts->trace) return vts->trace;

  Node *constant = state->graph->appendNode(state->graph->createConstant(var->data));
  constant->inferTypeFrom(var->data);
  setValueTrace(state, var, constant);
  return constant;
}

inline Node* getBufferTrace(const std::unordered_map<void*, Node*>& buffer_map, at::Tensor buf) {
  auto it = buffer_map.find(buf.unsafeGetTH(false));
  if (it == buffer_map.end()) {
    throw std::runtime_error("untraced buffer");
  } else {
    return it->second;
  }
}

// Only one field may be non-null
struct TraceInput {
  std::shared_ptr<Variable> variable;
  at::Tensor buffer;
  TraceInput(std::shared_ptr<Variable> variable) : variable(variable) {}
  TraceInput(at::Tensor buffer) : buffer(buffer) {}
  TraceInput() {}
};

// Start tracing, treating 'inputs' as inputs to the trace, which can be
// varied on subsequent invocations of the trace.  Any other variables
// will be treated as constants.
//
// NB: Why does this take an rvalue reference?  We need to get a non-const
// reference to at::Tensor buffer to call unsafeGetTH, but you can't get this
// out of a const vector (silly std::vector...)
inline std::shared_ptr<TracingState> enter(std::vector<TraceInput>&& trace_inputs, std::size_t num_stages) {
  auto state = std::make_shared<TracingState>(num_stages);
  variable_list inputs;
  for (auto& trace_input : trace_inputs) {
    if (trace_input.variable != nullptr) {
      JIT_ASSERT(!trace_input.buffer.defined());
      auto& input = trace_input.variable;
      Node *input_node = state->graph->addInput();
      setValueTrace(state, input, input_node);
      input_node->inferTypeFrom(input->data);
      inputs.push_back(input);
    } else {
      JIT_ASSERT(trace_input.buffer.defined());
      // NON-owning reference.  Pointers may be dead!
      auto& buffer = trace_input.buffer;
      Node* n = state->graph->addInput();
      state->buffer_map.insert({buffer.unsafeGetTH(false), n});
      n->inferTypeFrom(buffer);
    }
  }
  state->active = true;
  state->inputs = inputs;
  return state;
}

namespace detail {

// Exit code shared between exit and TraceExitHook::run
inline void _exit(const std::shared_ptr<TracingState>& state, const variable_list& outputs) {
  for (auto& output : outputs) {
    state->graph->registerOutput(getValueTrace(state, output, true));
  }
  state->active = false;
}

// Marks a backwards subgraph that should be traced as the next stage.
// Mutates some of the outputs.
void traceBackward(const std::shared_ptr<TracingState>& state, const variable_list& inputs,
                   const variable_list& outputs);

} // namespace detail

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
inline void exit(const variable_list& outputs) {
  auto state = getTracingState(outputs);
  detail::_exit(state, outputs);
  detail::traceBackward(state, state->inputs, outputs);
  state->inputs.clear();
}

// Marks part of the backward graph as non-traceable (i.e. one that should be replaced
// with an Eval in the trace).
void nontraceableBackwardSubgraph(const variable_list& inputs, const variable_list& outputs);

}}} // namespace torch::jit::tracer
