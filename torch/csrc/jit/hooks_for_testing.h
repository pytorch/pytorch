#pragma once
#include <functional>
#include "torch/csrc/WindowsTorchApiMacro.h"
#include <memory>

namespace torch {
namespace jit {
namespace script {
struct Module;
}
TORCH_API void didFinishEmitModule(std::shared_ptr<script::Module> module);
TORCH_API void setEmitModuleHook(std::function<void(std::shared_ptr<script::Module> module)> cb);
} // namespace jit
} // namespace torch
