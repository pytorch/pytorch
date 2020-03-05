#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/api/module.h>

#include <memory>

namespace torch {
namespace jit {

TORCH_API void UnpackQuantizedWeights(std::shared_ptr<Graph>& graph, std::map<std::string, at::Tensor>& paramsDict);
TORCH_API void insertPermutes(std::shared_ptr<Graph>& graph, std::map<std::string, at::Tensor>& paramsDict);
} // namespace jit
} // namespace torch
