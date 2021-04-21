#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// CAUTION NOT TO BE USED, STILL A WIP, NOT STABLE

TORCH_API void RegisterOperatorShapeFunction(
    Node* n,
    std::shared_ptr<Graph>& graph);

TORCH_API void PropagateShapesOnGraph(std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
