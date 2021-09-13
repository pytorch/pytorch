#pragma once

#include <ATen/core/function.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/api/include/torch/imethod.h>
#include <torch/csrc/jit/api/function_impl.h>

namespace torch {
namespace jit {

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;

// A method in a module, e.g. f in:
//
// class M(ScriptModule):
//   @script_method
//   def f(self, x):
//     ...
// Note: because Method/Module are exposed to python these
// classes use python method naming conventions
struct TORCH_API Method : public torch::IMethod {
  Method(ObjectPtr owner, Function* function);

  // the module that contains this method.
  Module owner() const;
  void run(Stack& stack);
  void run(Stack&& stack) {
    run(stack);
  }

  c10::IValue operator()(
      std::vector<c10::IValue> stack,
      const Kwargs& kwargs = Kwargs()) const override;

  // Run method async. Invocation on this function would invokes a JIT
  // interpreter that executes ops inline, one by one, on caller's thread. A
  // model can utilize async op, i.e. `fork`, to launch an asynchronous task
  // which will be launched on provided `taskLauncher`.
  c10::intrusive_ptr<c10::ivalue::Future> run_async(
      std::vector<c10::IValue> stack,
      const Kwargs& kwargs = Kwargs(),
      TaskLauncher taskLauncher = at::launch);

  std::shared_ptr<Graph> graph() const {
    return function_->graph();
  }

  const std::string& name() const override {
    return function_->name();
  }

  size_t num_inputs() const {
    return function_->num_inputs();
  }

  GraphExecutor& get_executor() {
    return function_->get_executor();
  }

  Function& function() const {
    return *function_;
  }

 private:
  void setArgumentNames(std::vector<std::string>&) const override;

  // Methods are uniqued onwed by a single module. This raw pointer allows
  // looking up the module.
  ObjectPtr owner_;

  // Underlying unbound function
  Function* function_;
};

namespace script {
// We once had a `script::` namespace that was deleted. This is for backcompat
// of the public API; new code should not use this type alias.
using Method = ::torch::jit::Method;
} // namespace script

} // namespace jit
} // namespace torch
