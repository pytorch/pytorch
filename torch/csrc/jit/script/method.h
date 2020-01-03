
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/function.h>

namespace torch {
namespace jit {
namespace script {

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;

// A method in a module, e.g. f in:
//
// class M(ScriptModule):
//   @script_method
//   def f(self, x):
//     ...
// Note: because Method/Module are exposed to python these
// classes use python method naming conventions
struct TORCH_API Method {
  Method(ObjectPtr owner, Function* function);

  // the module that contains this method.
  Module owner() const;
  void run(Stack& stack);
  void run(Stack&& stack) {
    run(stack);
  }

  c10::IValue operator()(
      std::vector<c10::IValue> stack,
      const Kwargs& kwargs = Kwargs());

  std::shared_ptr<Graph> graph() const {
    return function_->graph();
  }

  const std::string& name() const {
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
  // Methods are uniqued onwed by a single module. This raw pointer allows
  // looking up the module.
  ObjectPtr owner_;

  // Underlying unbound function
  Function* function_;
};

} // namespace script
} // namespace jit
} // namespace torch
