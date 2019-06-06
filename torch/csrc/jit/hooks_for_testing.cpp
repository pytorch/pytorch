#include <torch/csrc/jit/hooks_for_testing.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

static std::function<void(std::shared_ptr<script::Module> module)>
    emit_module_callback;
void didFinishEmitModule(std::shared_ptr<script::Module> module) {
  if (emit_module_callback) {
    emit_module_callback(std::move(module));
  }
}
static std::function<void(std::shared_ptr<script::Function> fn)>
    emit_function_callback;
void didFinishEmitFunction(std::shared_ptr<script::Function> fn) {
  if (emit_function_callback) {
    emit_function_callback(fn);
  }
}
void setEmitHooks(
    std::function<void(std::shared_ptr<script::Module> module)> for_mod,
    std::function<void(std::shared_ptr<script::Function> for_fn)> for_fn) {
  emit_module_callback = std::move(for_mod);
  emit_function_callback = std::move(for_fn);
}

} // namespace jit
} // namespace torch
