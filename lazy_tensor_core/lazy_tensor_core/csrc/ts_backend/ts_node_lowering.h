#pragma once

#include <torch/jit.h>
#include "lazy_tensor_core/csrc/lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {
using TSOpVector = std::vector<torch::jit::Value*>;

TSOpVector LowerTSBuiltin(
      std::shared_ptr<torch::jit::GraphFunction> function,
      c10::Symbol sym, const std::vector<torch::jit::NamedValue>& arguments,
      const std::vector<torch::jit::NamedValue>& kwarguments = {});

}  // namespace compiler
}  // namespace torch_lazy_tensors
