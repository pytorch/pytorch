#include <torch/csrc/jit/hooks_for_testing.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

static ModuleHook emit_module_callback;
void didFinishEmitModule(script::Module module) {
  if (emit_module_callback) {
    emit_module_callback(std::move(module));
  }
}

static FunctionHook emit_function_callback;
void didFinishEmitFunction(std::shared_ptr<Function> fn) {
  if (emit_function_callback) {
    emit_function_callback(fn);
  }
}

void setEmitHooks(ModuleHook for_mod, FunctionHook for_fn) {
  emit_module_callback = std::move(for_mod);
  emit_function_callback = std::move(for_fn);
}

std::pair<ModuleHook, FunctionHook> getEmitHooks() {
  return std::make_pair(emit_module_callback, emit_function_callback);
}

} // namespace jit
} // namespace torch
