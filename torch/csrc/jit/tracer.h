#pragma once

#include <ATen/Backtrace.h>
#include <ATen/core/functional.h>
#include <ATen/core/stack.h>
#include <c10/util/Exception.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/constants.h>
#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/tracing_state.h>
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

TORCH_API std::function<void()> pauseTracing();

TORCH_API Value* getValueTrace(const IValue& var);

TORCH_API Value* getNestedValueTrace(const IValue& v);

TORCH_API Value* getOutputTrace(
    const std::shared_ptr<TracingState>& state,
    const Variable& var);

TORCH_API Value* getNestedOutputTrace(
    const std::shared_ptr<TracingState>& state,
    const IValue& iv);

TORCH_API std::pair<std::shared_ptr<TracingState>, Stack> enter(Stack inputs);

TORCH_API void exit(const Stack& outputs);

TORCH_API void abandon();

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
TORCH_API void addInputs(Node* n, const char* name, at::IntArrayRef value);
TORCH_API void addInputs(
    Node* n,
    const char* name,
    at::TensorList value,
    bool allow_undefined = false);
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

TORCH_API void ensureUniqueIfOutOfPlaced(
    const char* name,
    const at::Tensor& tensor);

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
