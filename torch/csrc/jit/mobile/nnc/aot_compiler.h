#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/mobile/nnc/context.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

// Performs Ahead Of Time compilation of a given method in a model
// returning the compiled function and LLVM assembly code
TORCH_API std::unique_ptr<Function> aotCompile(
    const std::string& method_name,
    std::shared_ptr<Graph>& subgraph,
    const std::vector<int64_t>& sizes,
    std::string* compiled_assembly);

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
