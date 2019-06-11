#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <functional>
#include <memory>

namespace torch {
namespace jit {
namespace script {
struct Module;
}
TORCH_API void didFinishEmitModule(std::shared_ptr<script::Module> module);
TORCH_API void didFinishEmitFunction(StrongFunctionPtr defined);
TORCH_API void setEmitHooks(
    std::function<void(std::shared_ptr<script::Module> module)> for_module,
    std::function<void(StrongFunctionPtr)> for_fn);
} // namespace jit
} // namespace torch
