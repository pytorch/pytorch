#pragma once
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <functional>
#include <memory>

namespace torch {
namespace jit {
namespace script {
struct Module;
struct Function;
}
TORCH_API void didFinishEmitModule(std::shared_ptr<script::Module> module);
TORCH_API void didFinishEmitFunction(std::shared_ptr<script::Function> defined);
TORCH_API void setEmitHooks(
    std::function<void(std::shared_ptr<script::Module> module)> for_module,
    std::function<void(std::shared_ptr<script::Function> fn)> for_fn);
} // namespace jit
} // namespace torch
