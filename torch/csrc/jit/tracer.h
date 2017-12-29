#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable.h"

#include <memory>
#include <mutex>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>


namespace torch { namespace jit { namespace tracer {

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

std::string getPythonInterpreterStackTrace();

namespace detail {

inline ValueTracingStateElem* getValueState(const std::shared_ptr<TracingState>& state, const Variable& var, bool alloc = true) {
  for (auto it = var.tracing_state()->begin(); it != var.tracing_state()->end();) {
    auto ts = it->state.lock();
    // GC of invalidated tracing states
    if (!ts) {
      auto current_it = it++;
      var.tracing_state()->erase(current_it);
      continue;
    } else if (ts == state) {
      return &(*it);
    }
    ++it;
  }
  if (alloc) {
    var.tracing_state()->emplace_front();
    auto & vts = var.tracing_state()->front();
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

inline std::vector<VariableFlags> getVarFlags(const variable_list& vars) {
  return fmap(vars, &VariableFlags::of);
}

}


// Should a function which takes 'vars' as inputs be traced?
// It sufficies for ONE variable to be tracing: any "untraced" variables
// are treated as constants.
//
// TODO: This code lives in the hotpath; make sure it is fast
inline bool isTracingVar(const Variable& var) {
  if (!var.defined() || !var.tracing_state()) return false;
  return std::any_of(var.tracing_state()->begin(), var.tracing_state()->end(), detail::isElemActive);
}

inline bool isTracingVar(at::ArrayRef<Variable> vars) {
  // Reference to avoid refcount bump
  for (const Variable& var : vars) {
    if (isTracingVar(var)) return true;
  }
  return false;
}

// NB: Don't forget to forward declare your template calls: they are going to
// recursively call one another!
template<typename... Args> inline bool isTracing();
template<typename... Args> inline bool isTracing(const at::Tensor& x, Args... args);
template<typename... Args> inline bool isTracing(at::ArrayRef<at::Tensor> xs, Args... args);

template<typename... Args>
inline bool isTracing() {
  return false;
}

template<typename... Args>
inline bool isTracing(const at::Tensor& x, Args... args) {
  return isTracingVar(x) || isTracing(args...);
}

template<typename... Args>
inline bool isTracing(at::ArrayRef<at::Tensor> xs, Args... args) {
  for (const auto & x : xs) {
    if (isTracingVar(x)) return true;
  }
  return isTracing(args...);
}

// Retrieve the tracing state which a function applied with 'vars' should
// be recorded to.  Precondition: isTracing(vars) == true.  At the moment,
// we don't support mixing up variables from different traces; this code
// will need to be revisited if that ever becomes supported.
inline std::shared_ptr<TracingState> getTracingState(const variable_list& vars) {
  std::shared_ptr<TracingState> state;
  for (auto& var : vars) {
    if (!var.defined() || !var.tracing_state()) continue;
    for (auto & vts : *var.tracing_state()) {
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
inline void setValueTrace(const std::shared_ptr<TracingState>& state, const Variable& var, Value *value) {
  JIT_ASSERT(var.defined());
  auto vts = detail::getValueState(state, var);
  vts->trace = value;
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
inline Value* getValueTrace(const std::shared_ptr<TracingState>& state, const Variable& var) {
  if (!var.defined()) {
    Node *n = state->graph->createUndefined();
    return state->graph->appendNode(n)->output();
  }

  auto vts = detail::getValueState(state, var, true);
  if (vts->trace) return vts->trace;

  // HACK.  In an ideal world, buffers would be wrapped in variables, permitting
  // us to trace them just like we normally would.  In fact, internally, within
  // ATen, buffers get precisely this treatment.
  //
  // However, propagating this treatment would require us to do some fairly
  // disruptive changes to Python userland, where buffers are expected to be
  // passed around as plain tensors inside modules.  Some day we should do
  // this, but for now, we wrap all buffers in one-off Variables.  This means
  // they'll show up as constants when we trace.
  //
  // To deal with this, we cheat a little and consult the buffer map to
  // see if the wrapped tensor corresponds to a buffer.  If it does, use
  // that instead of making a constant.
  auto it = state->buffer_map.find(var.data().unsafeGetTH(false));
  if (it != state->buffer_map.end()) {
    return it->second;
  }

  Value *constant = state->graph->appendNode(state->graph->createConstant(var.data()))->output();
  constant->inferTypeFrom(var.data());
  setValueTrace(state, var, constant);
  return constant;
}

inline Value* getOutputTrace(const std::shared_ptr<TracingState>& state, const Variable& var, size_t output_no) {
  if (!var.defined()) {
    Node *n = state->graph->createUndefined();
    return state->graph->appendNode(n)->output();
  }

  auto vts = detail::getValueState(state, var, false);
  if (!vts) {
    std::ostringstream os;
    os << "output " << output_no << " of traced region did not have observable "
       << "data dependence with trace inputs; this probably indicates your program "
       << "cannot be understood by the tracer.";
    throw std::runtime_error(os.str());
  }
  return vts->trace;
}

// Only one field may be non-null
struct TraceInput {
  Variable variable;
  at::Tensor buffer;
  TraceInput(Variable variable) : variable(std::move(variable)) {}
  TraceInput(at::Tensor buffer) : buffer(std::move(buffer)) {}
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
    if (trace_input.variable.defined()) {
      JIT_ASSERT(!trace_input.buffer.defined());
      auto& input = trace_input.variable;
      auto input_node = state->graph->addInput(input.name());
      setValueTrace(state, input, input_node);
      input_node->inferTypeFrom(input.data());
      inputs.push_back(input);
    } else {
      JIT_ASSERT(trace_input.buffer.defined());
      // NON-owning reference.  Pointers may be dead!
      auto& buffer = trace_input.buffer;
      auto n = state->graph->addInput();
      state->buffer_map.insert({buffer.unsafeGetTH(false), n});
      n->inferTypeFrom(buffer);
    }
  }
  // TODO: this might not work with the way we handle buffers
  state->var_flags[0].first = detail::getVarFlags(inputs);
  state->active = true;
  state->inputs = inputs;
  return state;
}

namespace detail {

// Exit code shared between exit and TraceExitHook::run
inline void _exit(const std::shared_ptr<TracingState>& state, const variable_list& outputs) {
  size_t i = 0;
  for (auto& output : outputs) {
    state->graph->registerOutput(getOutputTrace(state, output, i));
    i++;
  }
  state->active = false;
  state->var_flags[state->graph->stage()].second = detail::getVarFlags(outputs);
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

Node* recordTrace(std::string op, at::ArrayRef<Variable> inputs, at::ArrayRef<Variable> outputs);

}}} // namespace torch::jit::tracer
