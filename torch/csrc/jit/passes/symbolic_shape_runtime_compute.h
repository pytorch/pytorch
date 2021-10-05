#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <unordered_map>

namespace torch {
namespace jit {

TORCH_API std::unordered_map<int64_t, Value*> InsertSymbolicShapesCompute(const ShapeComputeGraphMapping& shape_mapping, Node * insert_point);

// This is incomplete but currently it takes in a TensorExpr Graph, turns its non-1 dimensions into a symbolic shape,
// Propagates, and returns a map from Sym Shape Value to its runtime computed value
TORCH_API c10::optional<std::unordered_map<int64_t, Value*>>  GenerateGuard(Node * tensorexpr_graph_node);

} // namespace jit
} // namespace torch
