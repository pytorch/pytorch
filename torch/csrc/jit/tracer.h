#pragma once

#include "torch/csrc/autograd/function_hook.h"
#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/jit/assertions.h"
#include "torch/csrc/jit/constants.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/functional.h"
#include "torch/csrc/utils/variadic.h"
#include "torch/csrc/utils/variadic.h"
#include "torch/csrc/WindowsTorchApiMacro.h"

#include <ATen/Backtrace.h>

#include <memory>
#include <mutex>
#include <vector>
#include <iostream>
#include <cstdint>
#include <unordered_map>

namespace torch { namespace jit { namespace tracer {

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

TORCH_API void recordSourceLocation(Node* n);
TORCH_API void setRecordSourceLocation(void (*v)(Node*));

struct TORCH_API TracingState : public std::enable_shared_from_this<TracingState> {
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
      return t1.is_same(t2);
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

  TORCH_API static void stashIntListElem(const std::string& arg_name,
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
TORCH_API const std::shared_ptr<TracingState>& getTracingState();
TORCH_API void setTracingState(std::shared_ptr<TracingState> state);

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
    Value *constant = state->graph->insertConstant(var.data());
    recordSourceLocation(constant->node());
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
inline std::pair<std::shared_ptr<TracingState>, Stack> enter(Stack inputs) {
  if (isTracing()) {
    AT_ERROR("Tracing can't be nested");
  }
  auto state = std::make_shared<TracingState>();
  setTracingState(state);
  // XXX: this function mutates input
  const std::function<IValue(IValue, TypePtr, Value*)> add_input = [&](IValue input, TypePtr type, Value* value) -> IValue {
    value->setType(type);
    if (type->isSubtypeOf(DynamicType::get())) {
      auto input_tensor = input.toTensor();
      auto name = Variable(input_tensor).name();
      if (state->value_map.find(input_tensor) != state->value_map.end()) {
        input_tensor = input_tensor.view(input_tensor.sizes());
      }
      value->setUniqueName(name);
      state->value_map[input_tensor] = value;
      return input_tensor;
    } else if (auto tuple_type = type->cast<TupleType>()) {
      auto unpack_node = state->graph->insertNode(state->graph->createTupleUnpack(value));
      auto elem_values = unpack_node->outputs();
      auto elem_types = tuple_type->elements();
      Stack elems = input.toTuple()->elements();
      size_t num_elems = elems.size();
      AT_ASSERT(elem_values.size() == num_elems && elem_types.size() == num_elems);
      for (size_t i = 0; i < num_elems; ++i) {
        elems[i] = add_input(elems[i], elem_types[i], elem_values[i]);
      }
      return Tuple::create(std::move(elems));
    } else {
      AT_ERROR("Only tensors or tuples of tensors can be inputs to traced functions");
    }
  };
  for (IValue& input : inputs) {
    input = add_input(input, inferTypeFrom(input), state->graph->addInput());
  }
  return std::make_pair(state, inputs);
}

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
inline void exit(const Stack& outputs) {
  auto & state = getTracingState();
  size_t i = 0;
  std::function<Value*(const IValue&)> reduce_ivalue = [&](const IValue& iv) -> Value* {
    if (iv.isTensor()) {
      return getOutputTrace(state, iv.toTensor(), i);
    } else if (iv.isTuple()) {
      const auto & elems = iv.toTuple()->elements();
      auto tuple_node = state->graph->createTuple(fmap(elems, reduce_ivalue));
      state->graph->appendNode(tuple_node);
      return tuple_node->output();
    } else {
      AT_ERROR("Only tensors or tuples of tensors can be output from traced functions");
    }
  };
  for (auto& output : outputs) {
    state->graph->registerOutput(reduce_ivalue(output));
    i++;
  }
  setTracingState(nullptr);
}

// Abort tracing. Used to reset the state in case of errors.
inline void abandon() {
  setTracingState(nullptr);
}

// NB: those serve both as an intermediate steps in addInputs below,
// as well as the overloads that terminate template recursion
TORCH_API void addInputs(Node *n, const char * name, int64_t value);
TORCH_API void addInputs(Node *n, const char * name, bool value);
TORCH_API void addInputs(Node *n, const char * name, double value);
TORCH_API void addInputs(Node *n, const char * name, const at::Scalar& value);
TORCH_API void addInputs(Node *n, const char * name, const at::Tensor& value);
TORCH_API void addInputs(Node *n, const char * name, at::IntList value);
TORCH_API void addInputs(Node *n, const char * name, at::TensorList value);
TORCH_API void addInputs(Node *n, const char * name, const ArrayRef<double>& value);
TORCH_API void addInputs(Node *n, const char * name, const std::string& value);
TORCH_API void addInputs(Node *n, const char * name, const at::SparseTensorRef& value);
TORCH_API void addInputs(Node *n, const char * name, const at::TensorOptions& value);

template<size_t N>
void addInputs(Node *n, const char * name, std::array<bool, N> value) {
  throw std::runtime_error("Found an unsupported argument type in the JIT tracer. File a bug report.");
}

TORCH_API void postRecordTrace(Node* node, at::ArrayRef<Variable> outputs);

inline void postRecordTrace(Node* node, at::ArrayRef<at::Tensor> tensors) {
  postRecordTrace(node, fmap<Variable>(tensors));
}

template <
    typename T,
    typename = torch::enable_if_t<
        (!std::is_convertible<torch::decay_t<T>, ArrayRef<Variable>>::value &&
         !std::is_convertible<torch::decay_t<T>, ArrayRef<at::Tensor>>::value &&
         !std::is_convertible<torch::decay_t<T>, Variable>::value)>>
void postRecordTrace(Node* node, T&&) {
  AT_ERROR(
      "Found an unsupported argument type ", at::demangle_type<T>(),
      " in the JIT tracer. File a bug report.");
}

TORCH_API autograd::Variable getSizeOf(const autograd::Variable& var, int64_t dim);

}}} // namespace torch::jit::tracer
