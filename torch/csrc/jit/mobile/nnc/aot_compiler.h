#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/mobile/nnc/context.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace mobile {
namespace nnc {


// std::vector<int64_t>& ?
TORCH_API std::unique_ptr<Function> aot_compile(
    const std::string& method_name,
    std::shared_ptr<Graph>& subgraph,
    const std::vector<int64_t>& sizes,
    std::string* compiled_assembly);

}
}
}
}
