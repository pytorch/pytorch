#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <unordered_map>

namespace torch {
namespace jit {

TORCH_API std::unordered_map<int64_t, Value*> InsertSymbolicShapesCompute(const ShapeComputeGraphMapping& shape_mapping, Node * insert_point);

TORCH_API void GenerateGuard(Node * tensorexpr_graph_node);

} // namespace jit
} // namespace torch
