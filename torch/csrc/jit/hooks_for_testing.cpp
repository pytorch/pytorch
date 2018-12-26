#include <torch/csrc/jit/hooks_for_testing.h>
#include <torch/csrc/jit/script/module.h>

namespace torch {
namespace jit {

static std::function<void(std::shared_ptr<script::Module> module)>
    emit_module_callback;
TORCH_API void didFinishEmitModule(std::shared_ptr<script::Module> module) {
  if (emit_module_callback) {
    emit_module_callback(std::move(module));
  }
}
TORCH_API void setEmitModuleHook(
    std::function<void(std::shared_ptr<script::Module> module)> cb) {
  emit_module_callback = std::move(cb);
}
} // namespace jit
} // namespace torch
