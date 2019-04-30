#pragma once
#include <c10/util/Optional.h>
#include <memory>
#include <vector>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>

namespace at {
class Tensor;
}
namespace c10 {
struct IValue;
}
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
using Stack = std::vector<c10::IValue>;
using c10::ivalue::Future;
using c10::ivalue::Tuple;

struct TORCH_API Code {
  Code() : pImpl(nullptr) {}
  explicit Code(const std::shared_ptr<Graph>& graph);
  ~Code();

  const std::vector<GraphExecutor*>& grad_executors();

  explicit operator bool() const {
    return pImpl != nullptr;
  }

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
  InterpreterContinuation(InterpreterState state_, Stack stack_, bool grad_mode_enabled_)
      : state(state_), stack(std::move(stack_)), grad_mode_enabled(grad_mode_enabled_) {}

  void operator()();

 private:
  InterpreterState state;
  Stack stack;
  bool grad_mode_enabled;
};
} // namespace jit
} // namespace torch
