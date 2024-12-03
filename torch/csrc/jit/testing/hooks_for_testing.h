#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <functional>
#include <memory>

namespace torch::jit {
struct Module;

using ModuleHook = std::function<void(Module module)>;
using FunctionHook = std::function<void(StrongFunctionPtr function)>;

TORCH_API void didFinishEmitModule(Module module);
TORCH_API void didFinishEmitFunction(StrongFunctionPtr defined);
TORCH_API void setEmitHooks(ModuleHook for_module, FunctionHook for_fn);

TORCH_API std::pair<ModuleHook, FunctionHook> getEmitHooks();

} // namespace torch::jit
