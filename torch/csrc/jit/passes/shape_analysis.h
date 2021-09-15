#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>
#include <memory>

namespace torch {
namespace jit {

struct Graph;

TORCH_API void EraseShapeInformation(const std::shared_ptr<Graph>& graph);
TORCH_API void PropagateInputShapes(const std::shared_ptr<Graph>& graph);

TORCH_API bool mergeTypes(
    ArrayRef<Value*> lhs,
    ArrayRef<Value*> rhs,
    ArrayRef<Value*> outputs);

} // namespace jit
} // namespace torch
