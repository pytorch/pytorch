#pragma once

#include <torch/csrc/jit/ir.h>
#include <torch/csrc/jit/script/module.h>

#include <memory>

namespace torch {
namespace jit {

TORCH_API void UnpackQuantizedWeights(std::shared_ptr<Graph>& graph, std::map<std::string, at::Tensor>& paramsDict);
} // namespace jit
} // namespace torch
