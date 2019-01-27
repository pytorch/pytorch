#pragma once

#include <ATen/Backtrace.h>
#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/stack.h>
#include <torch/csrc/jit/tracing_state.h>
#include <torch/csrc/utils/functional.h>
#include <torch/csrc/utils/variadic.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tracer {

using ::c10::ivalue::List;
using ::c10::ivalue::Shared;

using ::c10::IValue;
using ::c10::ivalue::Future;
using ::c10::ivalue::Tuple;

using ::c10::ivalue::BoolList;
using ::c10::ivalue::DoubleList;
using ::c10::ivalue::GenericList;
using ::c10::ivalue::IntList;
using ::c10::ivalue::TensorList;

using ::c10::ivalue::ConstantString;

using torch::autograd::Variable;
using variable_list = std::vector<Variable>;

TORCH_API void recordSourceLocation(Node* n);
TORCH_API void setRecordSourceLocation(void (*v)(Node*));

// Having finished adding a new 'node' to the graph IR 'setValueTrace'
// associates this node with an output variable, so that further operations
// involving this variable know which node in the IR to reference.
TORCH_API void setValueTrace(const IValue& v, Value* value);

TORCH_API void delValueTrace(const Variable& var);

inline std::function<void()> pauseTracing() {
  std::shared_ptr<tracer::TracingState> state = getTracingState();
  tracer::setTracingState(nullptr);

  return [state]() { tracer::setTracingState(state); };
}

TORCH_API Value* getValueTrace(const IValue& var);

// allow tracing of tuples passed to List[Tensor] or Tuple[Tensor...] arguments
// One might merge getValueTrace and getNestedValueTrace after checking that
// casting to IValue instead  of Variable is OK
inline Value* getNestedValueTrace(const IValue& v) {
  auto& state = getTracingState();
  if (v.isTensorList()) {
    return state->graph
        ->insertNode(state->graph->createList(
            DynamicType::get(),
            fmap(
                v.toTensorListRef(),
                [](const IValue& val) { return getNestedValueTrace(val); })))
        ->output();
  } else if (v.isTuple()) {
    return state->graph
        ->insertNode(state->graph->createTuple(fmap(
            v.toTuple()->elements(),
            [](const IValue& val) { return getNestedValueTrace(val); })))
        ->output();
  }
  return getValueTrace(v.toTensor());
}

inline Value* getOutputTrace(
    const std::shared_ptr<TracingState>& state,
    const Variable& var) {
  if (!var.defined()) {
    Node* n = state->graph->createUndefined();
    return state->graph->insertNode(n)->output();
  }

  auto& value_map = getTracingState()->env_stack.back().value_map;
  auto it = value_map.find(var);
  if (it == value_map.end()) {
    std::ostringstream os;
    os << "output of traced region did not have observable "
       << "data dependence with trace inputs; this probably indicates your program "
       << "cannot be understood by the tracer.";
    throw std::runtime_error(os.str());
  }
  return it->second;
}

inline Value* getNestedOutputTrace(
    const std::shared_ptr<TracingState>& state,
    const IValue& iv) {
  if (iv.isTensor()) {
    return getOutputTrace(state, iv.toTensor());
  } else if (iv.isTuple()) {
    const auto& elems = iv.toTuple()->elements();
    auto tuple_node = state->graph->createTuple(
        fmap(elems, [&state](const IValue& iv) {
          return getNestedOutputTrace(state, iv);
        }));
    state->graph->insertNode(tuple_node);
    return tuple_node->output();
  } else {
    AT_ERROR(
        "Only tensors or tuples of tensors can be output from traced functions");
  }
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
  const std::function<IValue(IValue, TypePtr, Value*)> add_input =
      [&](IValue input, TypePtr type, Value* value) -> IValue {
    value->setType(type);
    if (type->isSubtypeOf(DynamicType::get())) {
      auto input_tensor = input.toTensor();
      auto name = Variable(input_tensor).name();
      auto& value_map = state->env_stack.back().value_map;
      if (value_map.find(input_tensor) != value_map.end()) {
        input_tensor = input_tensor.view(input_tensor.sizes());
      }
      value->setUniqueName(name);
      value_map[input_tensor] = value;
      return input_tensor;
    } else if (auto tuple_type = type->cast<TupleType>()) {
      auto unpack_node =
          state->graph->insertNode(state->graph->createTupleUnpack(value));
      auto elem_values = unpack_node->outputs();
      auto elem_types = tuple_type->elements();
      Stack elems = input.toTuple()->elements();
      size_t num_elems = elems.size();
      AT_ASSERT(
          elem_values.size() == num_elems && elem_types.size() == num_elems);
      for (size_t i = 0; i < num_elems; ++i) {
        elems[i] = add_input(elems[i], elem_types[i], elem_values[i]);
      }
      return Tuple::create(std::move(elems));
    } else {
      AT_ERROR(
          "Only tensors or tuples of tensors can be inputs to traced functions");
    }
  };
  for (IValue& input : inputs) {
    input = add_input(
        input, incompleteInferTypeFrom(input), state->graph->addInput());
  }
  return std::make_pair(state, inputs);
}

// Exit a trace, treating 'outputs' as the outputs of the trace.  These
// are the variables whose values will be computed upon subsequent
// invocations of the trace.
inline void exit(const Stack& outputs) {
  auto& state = getTracingState();
  size_t i = 0;
  for (auto& output : outputs) {
    state->graph->registerOutput(getNestedOutputTrace(state, output));
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
TORCH_API void addInputs(Node* n, const char* name, int64_t value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    c10::optional<int64_t> value);
TORCH_API void addInputs(Node* n, const char* name, bool value);
TORCH_API void addInputs(Node* n, const char* name, double value);
TORCH_API void addInputs(Node* n, const char* name, const at::Scalar& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::Scalar>& value);
TORCH_API void addInputs(Node* n, const char* name, const at::Tensor& value);
TORCH_API void addInputs(Node* n, const char* name, at::IntList value);
TORCH_API void addInputs(Node* n, const char* name, at::TensorList value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const ArrayRef<double>& value);
TORCH_API void addInputs(Node* n, const char* name, const std::string& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const at::SparseTensorRef& value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const at::TensorOptions& value);
TORCH_API void addInputs(Node* n, const char* name, at::Device value);
TORCH_API void addInputs(Node* n, const char* name, at::Layout value);
TORCH_API void addInputs(Node* n, const char* name, at::ScalarType value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    const c10::optional<at::ScalarType>& value);
TORCH_API void addInputs(Node* n, const char* name, at::Generator* value);

template <size_t N>
void addInputs(Node* n, const char* name, std::array<bool, N> value) {
  throw std::runtime_error(
      "Found an unsupported argument type in the JIT tracer. File a bug report.");
}

inline void ensureUniqueIfOutOfPlaced(
    const char* name,
    const at::Tensor& tensor) {
  auto& state = getTracingState();
  if (state && state->force_outplace == false) {
    // If we're not converting in-place ops to out-of-place, this check is
    // unnecessary
    return;
  }
  auto aliases = tensor.storage().use_count();
  if (isTracing() && aliases > 1) {
    std::stringstream ss;
    ss << "There are " << aliases
       << " live references to the data region being modified when tracing in-place operator "
       << name
       << ". This might cause the trace to be incorrect, because all other views "
       << "that also reference this data will not not reflect this change in the trace! "
       << "On the other hand, if all other views use the same memory chunk, but are disjoint (e.g. "
       << "are outputs of torch.split), this might still be safe.";
    warn(ss.str().c_str());
  }
}

template <
    typename T,
    typename = torch::enable_if_t<
        (!std::is_convertible<torch::decay_t<T>, at::TensorList>::value &&
         !std::is_convertible<torch::decay_t<T>, at::Tensor>::value)>>
void addOutput(Node* node, T&&) {
  AT_ERROR(
      "Found an unsupported argument type ",
      c10::demangle_type<T>(),
      " in the JIT tracer. File a bug report.");
}
TORCH_API void addOutput(Node* node, const at::Tensor& tensor);
TORCH_API void setOutput(Value* value, const at::Tensor& output);
TORCH_API void addOutput(Node* node, const std::vector<at::Tensor>& list);

TORCH_API autograd::Variable getSizeOf(
    const autograd::Variable& var,
    int64_t dim);

} // namespace tracer
} // namespace jit
} // namespace torch
