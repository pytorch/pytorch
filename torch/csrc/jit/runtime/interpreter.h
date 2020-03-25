#pragma once
#include <c10/util/Optional.h>
#include <memory>
#include <vector>

#include <ATen/core/ivalue.h>
#include <torch/csrc/ThreadLocalState.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace at {
class Tensor;
}
namespace c10 {
struct IValue;
struct OperatorName;
} // namespace c10

namespace torch {
namespace jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.

struct Node;
struct GraphExecutor;
struct CodeImpl;
struct InterpreterStateImpl;
struct Graph;
struct Node;
struct Instruction;
using Stack = std::vector<c10::IValue>;
using c10::ivalue::Future;

struct TORCH_API Code {
  Code() : pImpl(nullptr) {}
  // remaining_bailout_depth is irrelevant in a `Code` object unless the `Code`
  // is directly created by `GraphExecutor` in which case it's likely to contain
  // `prim::BailOut`s to control the maximum depth of bailout chains
  explicit Code(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      size_t remaining_bailout_depth = 0);
  ~Code();

  const std::vector<GraphExecutor*>& grad_executors();

  explicit operator bool() const {
    return pImpl != nullptr;
  }
  size_t num_inputs() const;
  size_t num_outputs() const;
  size_t num_bailouts() const;
  const std::vector<c10::IValue>& constant_table() const;
  const std::vector<c10::TypePtr>& type_table() const;
  const std::vector<Instruction>& instructions() const;
  const std::vector<Node*>& instructions_source() const;
  void request_bailout(size_t index);
  size_t register_size() const;
 private:
  std::shared_ptr<CodeImpl> pImpl;
  friend struct InterpreterStateImpl;
  friend std::ostream& operator<<(std::ostream& out, const Code& code);
};

struct InterpreterState {
  TORCH_API InterpreterState(const Code& code);
  TORCH_API void run(Stack& stack);
  c10::intrusive_ptr<Future> runAsync(Stack& stack);
  c10::intrusive_ptr<Future> getFuture();
  TORCH_API ~InterpreterState();

 private:
  InterpreterState(c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl);
  // Ideally we should use c10::intrusive_ptr<InterpreterStateImpl> for pImpl;
  // but intrusive_ptr requires full definition of InterpreterStateImpl,
  // which we need to hide in the header.
  c10::intrusive_ptr<c10::intrusive_ptr_target> pImpl;
  friend struct InterpreterStateImpl;
};

// Created by wait()
struct Suspend : public std::exception {
  const char* what() const noexcept override {
    return "Suspend";
  }

  explicit Suspend(c10::intrusive_ptr<Future> future_)
      : future(std::move(future_)) {}

  c10::intrusive_ptr<Future> future;
};

struct InterpreterContinuation {
  InterpreterContinuation(
      InterpreterState state_,
      Stack stack_,
      torch::ThreadLocalState thread_local_state_)
      : state(state_),
        stack(std::move(stack_)),
        thread_local_state(std::move(thread_local_state_)) {}

  void operator()();

 private:
  InterpreterState state;
  Stack stack;
  // Thread local state that needs to be carried over to the continuation.
  torch::ThreadLocalState thread_local_state;
};

// what is the tensors type, including state from the current execution context
// that modifies how the tensor behaves. For instance if no_grad is enabled
// this will cause the TensorType to have requires_grad=False.
TORCH_API at::TensorTypePtr tensorTypeInCurrentExecutionContext(
    const at::Tensor& t);

} // namespace jit
} // namespace torch
