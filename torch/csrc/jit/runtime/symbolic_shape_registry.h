#pragma once
// This file is temporary until native_functions.yaml and derivatives.yaml are
// merged. Ideally this should all go into native_functions.yaml

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

TORCH_API c10::optional<std::shared_ptr<Graph>> shapeComputeGraphForOperator(
    const Operator& op);

TORCH_API void registerShapeFunction(const std::shared_ptr<Operator>& op, std::shared_ptr<Graph> graph);

} // namespace jit
} // namespace torch
