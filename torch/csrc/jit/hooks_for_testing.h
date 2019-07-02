#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <functional>
#include <memory>

namespace torch {
namespace jit {
struct Function;
namespace script {
struct Module;
}

using ModuleHook = std::function<void(script::Module module)>;
using FunctionHook = std::function<void(std::shared_ptr<Function> function)>;

TORCH_API void didFinishEmitModule(script::Module module);
TORCH_API void didFinishEmitFunction(std::shared_ptr<Function> defined);
TORCH_API void setEmitHooks(
    ModuleHook for_module,
    FunctionHook for_fn);

TORCH_API std::pair<ModuleHook, FunctionHook> getEmitHooks();

} // namespace jit
} // namespace torch
