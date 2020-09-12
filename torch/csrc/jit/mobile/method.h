#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/mobile/function.h>

namespace torch {
namespace jit {
namespace mobile {

class Module;

struct TORCH_API Method {
  Method(const Module* owner, Function* function);

  void run(Stack& stack);
  void run(Stack&& stack) {
    run(stack);
  }

  c10::IValue operator()(std::vector<c10::IValue> stack);

  const std::string& name() const {
    return function_->name();
  }

  std::string get_module_debug_info(size_t pc) const {
    return function_->get_module_debug_info(pc);
  }

  Function& function() const {
    return *function_;
  }

 private:
  // Methods are uniquely owned by a single module.
  // This raw pointer allows referencing the module
  const Module* owner_;

  // Underlying unbound function
  Function* function_;
};

} // namespace mobile
} // namespace jit
} // namespace torch
