#pragma once

#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>

#include <memory>

namespace torch {
namespace jit {

TORCH_API void UnpackQuantizedWeights(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict);
TORCH_API void insertPermutes(
    std::shared_ptr<Graph>& graph,
    std::map<std::string, IValue>& paramsDict);
} // namespace jit
} // namespace torch
