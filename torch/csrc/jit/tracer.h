#pragma once

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/assertions.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/variadic.h"
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

struct TracingState : public std::enable_shared_from_this<TracingState> {
  TracingState();
  ~TracingState();

  using WeakTensor = at::WeakTensor;

  struct WeakTensorHasher {
    size_t operator()(const WeakTensor& t) const {
      return std::hash<void*>()(t.unsafeGetTensorImpl());
    }
  };

  struct WeakTensorEq {
    bool operator()(const WeakTensor& t1, const WeakTensor& t2) const {
      return t1.unsafeGetTensorImpl() == t2.unsafeGetTensorImpl();
    }
  };

  std::unordered_map<WeakTensor, Value*, WeakTensorHasher, WeakTensorEq> value_map;
  std::shared_ptr<Graph> graph;
};


// This is meant to be used as a thread local place, where we can store extra
// info that gets lost when we call into ATen from Python bindings. One example
// for when this happens is when we get an IntList argument with e.g. sizes for
// view. When tracing, those might be tensors, which let us encode extra data
// dependencies, but once they get to the ATen call where we actually have the
// tracing logic, they get converted into a raw IntList, and we loose all
// information. To prevent this, we temporarily stash it in here.
struct ArgumentStash {
  struct IntListTrace : std::vector<Value*> {
    IntListTrace(int size)
      : std::vector<Value*>(size, nullptr) {}
  };

  static bool empty() {
    return stash.intlists.empty();
  }

  static void stashIntListElem(const std::string& arg_name,
                               size_t size,
                               size_t idx,
                               const Variable& var);

  static bool hasIntList(const std::string& arg_name) {
    return stash.intlists.count(arg_name) > 0;
  }

  static IntListTrace popIntList(const std::string& arg_name) {
    auto info = std::move(stash.intlists.at(arg_name));
    stash.intlists.erase(arg_name);
    return info;
  }

private:
  static thread_local ArgumentStash stash;
  std::unordered_map<std::string, IntListTrace> intlists;
};

// Retrieve or set the current tracing state. Returns a nullptr if tracing is disabled.
const std::shared_ptr<TracingState>& getTracingState();
void setTracingState(std::shared_ptr<TracingState> state);

inline bool isTracing() {
  return static_cast<bool>(getTracingState());
}

// Having finished adding a new 'node' to the graph IR 'setValueTrace' associates
// this node with an output variable, so that further operations involving this
// variable know which node in the IR to reference.
inline void setValueTrace(const Variable& var, Value *value) {
  JIT_ASSERT(var.defined());
  getTracingState()->value_map[var] = value;
}

// Given a variable 'var', return the 'node' which represents the instruction
// which computes the value of this variable in the IR.
// Here, we interpret untraced variables as constants that are just embedded
// in the graph.  This is useful to handle code which does things like this
// (from torch.autograd.variable, now moved to C++):
//
//    def mm(self, matrix):
//      output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
//      return Addmm.apply(output, self, matrix, 0, 1, True)
//
// Here, mm fakes up a dummy variable with uninitialized data to do an inplace
// update on, but subsequently ignores it because the alpha scaling factor is zero.
// This is one of the cases where a Variable can be created inside of a trace, and
// if we treat it as a constant, everything will work out.
inline Value* getValueTrace(const Variable& var) {
  auto &state = getTracingState();
  if (!var.defined()) {
    Node *n = state->graph->createUndefined();
    return state->graph->appendNode(n)->output();
  }

  auto & value_map = getTracingState()->value_map;
  auto it = value_map.find(var);
  if (it == value_map.end()) {
    Value *constant = state->graph->appendNode(state->graph->createConstant(var.data()))->output();
    constant->inferTypeFrom(var.data());
    it = value_map.emplace_hint(it, var, constant);
  }
  return it->second;
}

inline Value* getOutputTrace(const std::shared_ptr<TracingState>& state, const Variable& var, size_t output_no) {
  if (!var.defined()) {
    Node *n = state->graph->createUndefined();
    return state->graph->appendNode(n)->output();
  }

  auto & value_map = getTracingState()->value_map;
  auto it = value_map.find(var);
  if (it == value_map.end()) {
    std::ostringstream os;
    os << "output " << output_no << " of traced region did not have observable "
       << "data dependence with trace inputs; this probably indicates your program "
       << "cannot be understood by the tracer.";
    throw std::runtime_error(os.str());
  }
  return it->second;
}

// Start tracing, treating 'inputs' as inputs to the trace, which can be
// varied on subsequent invocations of the trace.  Any other variables
// will be treated as constants.
inline std::pair<std::shared_ptr<TracingState>, variable_list> enter(
    variable_list inputs) {
  if (isTracing()) {
    AT_ERROR("Tracing can't be nested");
  }
  auto state = std::make_shared<TracingState>();
  setTracingState(state);
  for (auto& input : inputs) {
    auto * value_state = state->value_map[input];
    if (value_state) {
      // See Note [Repeated inputs] in tracer.cpp
      input = input.view(input.sizes());
    }
    auto input_node = state->graph->addInput(input.name());
    input_node->inferTypeFrom(input.data());
    state->value_map[input] = input_node;
  }
  return std::make_pair(state, inputs);
}

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
inline void exit(const variable_list& outputs) {
  auto & state = getTracingState();
  size_t i = 0;
  for (auto& output : outputs) {
    state->graph->registerOutput(getOutputTrace(state, output, i));
    i++;
  }
  setTracingState(nullptr);
}

// Abort tracing. Used to reset the state in case of errors.
inline void abandon() {
  setTracingState(nullptr);
}

// Pre-recorded information about the trace before we actually carry
// out the trace
struct PreTraceInfo {
  Node *n;
};

PreTraceInfo preRecordTrace(Symbol op, at::ArrayRef<Variable> inputs);
void postRecordTrace(const PreTraceInfo& info, at::ArrayRef<Variable> outputs);

void recordSourceLocation(Node* n);
void setRecordSourceLocation(void (*v)(Node*));

// We must record the nodes of inputs before we actually carry out
// the operation, because an inplace operation may destroy the information
// we're interested in.  See #4480.
template<typename F>
PreTraceInfo makePreTraceInfo(at::ArrayRef<Variable> inputs, F ctor) {
  PreTraceInfo info;
  auto & state = getTracingState();
  auto & graph = state->graph;

  Node *n = ctor(state, *graph);
  recordSourceLocation(n);

  for (const Variable & input : inputs) {
    n->addInput(getValueTrace(input));
  }

  // NB: Order matters. This must append after inputs but before outputs.
  graph->appendNode(n);

  info.n = n;

  return info;
}

autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim);

}}} // namespace torch::jit::tracer
