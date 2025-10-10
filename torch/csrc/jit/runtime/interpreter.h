#pragma once
#include <memory>
#include <optional>
#include <vector>

#include <ATen/ThreadLocalState.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/source_range.h>

TORCH_DECLARE_bool(torch_jit_disable_warning_prints);
TORCH_DECLARE_bool(torch_jit_enable_rethrow_caught_exception);

namespace at {
class Tensor;
TORCH_API void launch(std::function<void()> func);
} // namespace at
namespace c10 {
struct IValue;
struct OperatorName;
} // namespace c10

namespace torch::jit {

// The interpreter run Graphs with Tensor inputs and Tensor outputs
// a separate component in the autograd handles unwrapping and wrapping
// variable objects for use in the interpreter.
namespace interpreter {
struct CodeImpl;
}

struct Node;
struct GraphExecutor;
struct InterpreterStateImpl;
struct Graph;
struct Node;
struct Instruction;
using Stack = std::vector<c10::IValue>;
using c10::ivalue::Future;
using TaskLauncher = std::function<void(std::function<void()>)>;

bool TORCH_API in_torchscript_runtime();

struct TORCH_API Code {
  Code() = default;
  explicit Code(interpreter::CodeImpl* pImpl);
  // remaining_bailout_depth is irrelevant in a `Code` object unless the `Code`
  // is directly created by `GraphExecutor` in which case it's likely to contain
  // `prim::BailOut`s to control the maximum depth of bailout chains
  explicit Code(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      size_t remaining_bailout_depth = 0);

  const std::vector<GraphExecutor*>& grad_executors();
  const std::vector<GraphExecutor*>& diff_graph_op_executors();

  explicit operator bool() const {
    return pImpl != nullptr;
  }
  size_t num_inputs() const;
  size_t num_outputs() const;
  size_t num_bailouts() const;
  const std::vector<c10::IValue>& constant_table() const;
  const std::vector<c10::TypePtr>& type_table() const;
  const std::vector<Instruction>& instructions() const;
  const std::unordered_map<std::string, size_t>& op_to_num_specified_args()
      const;
  const std::vector<Node*>& instructions_source() const;
  void request_bailout(size_t index);
  size_t register_size() const;
  std::shared_ptr<Graph> graph() const;

 private:
  std::shared_ptr<interpreter::CodeImpl> pImpl;
  friend struct InterpreterStateImpl;
  friend std::ostream& operator<<(std::ostream& out, const Code& code);
};

struct TORCH_API MobileCode : Code {
  explicit MobileCode(
      const std::shared_ptr<Graph>& graph,
      std::string function_name,
      bool emit_default_input_instructions = true,
      bool support_default_args_before_out = true,
      bool emit_promoted_ops = true,
      size_t remaining_bailout_depth = 0);
};

struct InterpreterState {
  TORCH_API InterpreterState(
      const Code& code,
      TaskLauncher taskLauncher = at::launch);
  TORCH_API void run(Stack& stack);
  TORCH_API c10::intrusive_ptr<Future> runAsync(Stack& stack);
  c10::intrusive_ptr<Future> getFuture();

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

// InterpreterContinuation propagates dist_autograd_context_id
// through (and only through) the forward pass manually, other
// thread local settings are propagated with ThreadLocalState
struct InterpreterContinuation {
  InterpreterContinuation(
      InterpreterState state_,
      Stack stack_,
      int64_t dist_autograd_context_id = 0,
      std::optional<at::ThreadLocalState> tls_state = std::nullopt)
      : state(std::move(state_)),
        stack(std::move(stack_)),
        tls_state_(std::move(tls_state))
#ifdef USE_DISTRIBUTED
        ,
        dist_autograd_context_id_(dist_autograd_context_id)
#endif
  {
  }

  void operator()();

 private:
  InterpreterState state;
  Stack stack;
  std::optional<at::ThreadLocalState> tls_state_ = std::nullopt;
#ifdef USE_DISTRIBUTED
  int64_t dist_autograd_context_id_;
#endif
};

// what is the tensors type, including state from the current execution context
// that modifies how the tensor behaves. For instance if no_grad is enabled
// this will cause the TensorType to have requires_grad=False.
TORCH_API at::TensorTypePtr tensorTypeInCurrentExecutionContext(
    const at::Tensor& t);

// current (TLS) TorchScript interpreter callstack
TORCH_API std::vector<StackEntry> currentCallstack();
TORCH_API std::vector<std::string> currentModuleHierarchy();

} // namespace torch::jit
