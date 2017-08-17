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

using torch::autograd::Variable;
using variable_list = std::vector<std::shared_ptr<Variable>>;

// TracingState tracks the necessary state when we are tracing the execution of
// autograd code; most importantly, it holds a reference to the actual IR
// graph which we are recording the trace to.
//
// The liveness of a TracingState is expected to be a superset of the region
// of code being traced; in particular, Variables do not keep a TracingState
// live.  Instead, they hold weak pointers to TracingState, to prevent leaks
// from arising when a variable that participated in a trace outlives the
// actual trace itself.
//
// Multi-threaded tracing is NOT yet supported: it is better to think of
// tracing as a thread local affair, where we enter and then exit
// a tracing region.  This not only coincides with how Python code
// is run (GIL = single-threaded), but also how single Functions in
// an autograd closure are applied.  Note that the execution of an
// *entire* autograd closure is multithreaded, in which case extra
// locking is necessary.

struct TracingState : public std::enable_shared_from_this<TracingState> {
  TracingState()
    : graph(new Graph())
    , active(false) {}

  std::unique_ptr<Graph> graph;
  bool active;
};

namespace detail {

template<typename Subclass>
struct TracerHook : public autograd::FunctionPreHook {
protected:
  // Returns a vector of hooks that were registered. Subclasses can then perform additional initialization.
  static std::shared_ptr<Subclass> registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& inputs);

  std::shared_ptr<TracingState> tracing_state;

public:
  virtual void run(variable_list& inputs) = 0;

  // Handle both kinds of hooks. In case of post hooks we only care about outputs.
  virtual variable_list operator()(const variable_list& _vars) {
    variable_list vars(_vars);
    for (auto& var : _vars)
      JIT_ASSERT(var);
    using this_type = typename std::remove_reference<decltype(*this)>::type;
    std::call_once(flag, std::bind(&this_type::run, this, vars));
    JIT_ASSERT(vars.size() == _vars.size());
    for (auto& var : vars) {
      JIT_ASSERT(var);
    }
    return vars;
  }

private:
  std::once_flag flag;
};

////////////////////////////////////////////////////////////////////////////////
// Trace hooks
////////////////////////////////////////////////////////////////////////////////

struct TraceEnterHook : public TracerHook<TraceEnterHook> {
private:
  friend struct TracerHook<TraceEnterHook>;

  virtual void run(variable_list& inputs) override;

public:
  static void registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& outputs);
};

struct TraceExitHook : public TracerHook<TraceExitHook> {
private:
  friend struct TracerHook<TraceExitHook>;

  virtual void run(variable_list& outputs) override;

public:
  static void registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& inputs);
};

} // namespace detail

// Should a function which takes 'vars' as inputs be traced?
// It sufficies for ONE variable to be tracing: any "untraced" variables
// are treated as constants.
inline bool isTracing(const variable_list& vars) {
  for (auto& var : vars) {
    if (!var) continue;
    auto state = var->tracing_state.state.lock();
    if (state && state->active)
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

// Having finished adding a new 'node' to the graph IR owned by TracingState 'state',
// 'setValueTrace' associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
inline void setValueTrace(const std::shared_ptr<TracingState>& state, const std::shared_ptr<Variable>& var, Node *node) {
  JIT_ASSERT(var->tracing_state.state.lock() == state || var->tracing_state.state.expired());
  var->tracing_state.state = state;
  var->tracing_state.trace = node;
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
  JIT_ASSERTM(var, "Not supported. NULL Variables will need to be removed from autograd");
  auto var_state = var->tracing_state.state.lock();
  if (var_state) {
    JIT_ASSERT(var->tracing_state.state.lock() == state);
    return var->tracing_state.trace;
  }

  if (mustExist) throw std::runtime_error("untraced variable");

  Node *constant = state->graph->appendNode(state->graph->createConstant(var->data));
  setValueTrace(state, var, constant);
  return constant;
}

// Start tracing, treating 'inputs' as inputs to the trace, which can be
// varied on subsequent invocations of the trace.  Any other variables
// will be treated as constants.
// XXX: this changes variables in inputs!
inline std::shared_ptr<TracingState> enter(variable_list& inputs) {
  auto state = std::make_shared<TracingState>();
  for (auto& input : inputs) {
    JIT_ASSERT(input->tracing_state.state.expired());
    input->tracing_state.state = state;
    input->tracing_state.trace = state->graph->addInput();
    input->tracing_state.trace->inferTypeFrom(input->data);
  }
  state->active = true;
  detail::TraceExitHook::registerHook(state, inputs);
  return state;
}

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
inline void exit(variable_list& outputs) {
  auto state = getTracingState(outputs);
  for (auto& output : outputs) {
    state->graph->registerOutput(getValueTrace(state, output, true));
  }
  state->active = false;
  detail::TraceEnterHook::registerHook(state, outputs);
}

////////////////////////////////////////////////////////////////////////////////
// Eval hooks
////////////////////////////////////////////////////////////////////////////////

struct EvalCommonState {
  // Filled in by EvalEnterHook when ran
  Node* eval_node;
  std::shared_ptr<EvalCommonState> next_common_state;
};

struct EvalEnterHook : public detail::TracerHook<EvalEnterHook> {
private:
  friend detail::TracerHook<EvalEnterHook>;

  std::shared_ptr<EvalCommonState> common_state;

  virtual void run(variable_list& vars) override;

public:
  static void registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& outputs, std::shared_ptr<EvalCommonState> common_state);
};

struct EvalExitHook : public detail::TracerHook<EvalExitHook> {
private:
  friend detail::TracerHook<EvalExitHook>;

  std::shared_ptr<EvalCommonState> common_state;

  virtual void run(variable_list& vars) override;

public:
  static std::shared_ptr<EvalCommonState> registerHook(const std::shared_ptr<TracingState>& tracing_state, variable_list& inputs);
};

}}} // namespace torch::jit::tracer
