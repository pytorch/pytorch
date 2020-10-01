#include <torch/csrc/jit/testing/hooks_for_testing.h>
#include <torch/csrc/jit/api/module.h>

namespace torch {
namespace jit {

static ModuleHook emit_module_callback;
void didFinishEmitModule(Module module) {
  if (emit_module_callback) {
    emit_module_callback(module);
  }
}

static FunctionHook emit_function_callback;
void didFinishEmitFunction(StrongFunctionPtr fn) {
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
