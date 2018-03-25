#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/tracer_state.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/variadic.h"
#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/utils/auto_unique_ptr.h"
#ifndef NO_PYTHON
#include "torch/csrc/utils/pybind.h"
#endif
#include <memory>
#include <mutex>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>

namespace torch { namespace jit { namespace tracer {

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

#ifndef NO_PYTHON
std::string getPythonInterpreterStackTrace();
#endif

namespace detail {

inline ValueTracingStateElem* getValueState(const std::shared_ptr<TracingState>& state, const Variable& var, bool alloc = true) {
  auto& tracing_state = var.tracing_state();
  for (auto it = tracing_state.begin(); it != tracing_state.end();) {
    auto ts = it->state.lock();
    // GC of invalidated tracing states
    if (!ts) {
      auto current_it = it++;
      tracing_state.erase(current_it);
      continue;
    } else if (ts == state) {
      return &(*it);
    }
    ++it;
  }
  if (alloc) {
    tracing_state.emplace_front();
    auto & vts = tracing_state.front();
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
// It suffices for ONE variable to be tracing: any "untraced" variables
// are treated as constants.
//
// NB: This code lives in the hotpath; make sure it is fast
//
// NB: Variable overload is not variadic because we don't actually
// need it (in most cases if we have a variable_list it is already
// flattened).
inline bool isTracingVar(const Variable& var) {
  if (!var.defined() || !var.has_tracing_state()) return false;
  return std::any_of(var.tracing_state().begin(), var.tracing_state().end(), detail::isElemActive);
}

inline bool isTracingVar(at::ArrayRef<Variable> vars) {
  // Reference to avoid refcount bump
  for (const Variable& var : vars) {
    if (isTracingVar(var)) return true;
  }
  return false;
}

struct IsTracing : IterArgs<IsTracing> {
  bool out = false;
  using IterArgs<IsTracing>::operator();
  void operator()(const at::Tensor& var) {
    out = out || isTracingVar(var);
  }
  bool short_circuit() { return out; }
};

// To be called with Tensor arguments from generated code
template<typename... Args>
inline bool isTracing(Args&&... args) {
  return IsTracing().apply(std::forward<Args>(args)...).out;
}

// Retrieve the tracing state which a function applied with 'vars' should
// be recorded to.  Precondition: isTracing(vars) == true.  At the moment,
// we don't support mixing up variables from different traces; this code
// will need to be revisited if that ever becomes supported.
inline std::shared_ptr<TracingState> getTracingState(const variable_list& vars) {
  std::shared_ptr<TracingState> state;
  for (auto& var : vars) {
    if (!var.defined() || !var.has_tracing_state()) continue;
    for (auto & vts : var.tracing_state()) {
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

// Start tracing, treating 'inputs' as inputs to the trace, which can be
// varied on subsequent invocations of the trace.  Any other variables
// will be treated as constants.
//
// NB: Why does this take an rvalue reference?  We need to get a non-const
// reference to at::Tensor buffer to call unsafeGetTH, but you can't get this
// out of a const vector (silly std::vector...)
inline std::pair<std::shared_ptr<TracingState>, variable_list> enter(
    variable_list inputs, std::size_t num_stages) {
  auto state = std::make_shared<TracingState>(num_stages);
  for (auto& input : inputs) {
    auto * value_state = detail::getValueState(state, input, false);
    if (value_state) {
      // See Note [Repeated inputs] in tracer.cpp
      input = input.view(input.sizes());
    }
    auto input_node = state->graph->addInput(input.name());
    setValueTrace(state, input, input_node);
    input_node->inferTypeFrom(input.data());
  }
  state->var_flags[0].first = detail::getVarFlags(inputs);
  state->active = true;
  state->inputs = inputs;
  return std::make_pair(state, inputs);
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

// Pre-recorded information about the trace before we actually carry
// out the trace
struct PreTraceInfo {
  std::shared_ptr<TracingState> state;
  Node *n;
};

PreTraceInfo preRecordTrace(Symbol op, at::ArrayRef<Variable> inputs);
#ifndef NO_PYTHON
PreTraceInfo preRecordPythonTrace(
    THPObjectPtr pyobj, std::string arg_types, at::ArrayRef<Variable> inputs,
    pyobj_list scalar_args);

std::shared_ptr<Graph> createGraphByTracing(
        py::function func,
        variable_list inputs,
        size_t num_inputs);
#endif
void postRecordTrace(const PreTraceInfo& info, at::ArrayRef<Variable> outputs);




}}} // namespace torch::jit::tracer
