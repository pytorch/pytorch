#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <unordered_map>

namespace torch {
namespace jit {

// Takes in a TensorExprGraph of static shapes and generalizes the input shapes
// to symbolic dimensions. Dimensions of value 1 will be preserved, otherwise
// dimensions with the same value will be bucketed to the same symbolic shape.
// E.g. Tensor(5, 3), Tensor(3, 1) -> Tensor(SS(-1), SS(-2)), Tensor(SS(-2), 1)
// From there, runs symbolic shape inference on the graph, and creates a
// versionining if in the graph with prim::TensorExprDynamicGuard checking if
// the inputs at runtime match the Generalized Symbolic Shapes that are inputs
// to the TE Kernel. The computate to calculate all symbolic dimensions is
// inlined in to the if block with the TE Kernel. All Sym Dim Value* are
// appended to the end of the TE Kernel Graph/Node inputs, and the Node is
// augmented with a integer list attr `symbolic_shape_inputs` that gives the
// mapping from Value * -> Symbolic Shape int64_t value. For more lengthy IR
// examples and walkthrough look at ShapeAnalysisTest.DynamicShapesFusion in
// `test_shape_analysis` Returns True on Success, False on Failure, can fail if
// shape propagation fails to propagate # of dims or if complete shapes on
// inputs not set

TORCH_API bool GenerateGuard(
    Node* tensorexpr_graph_node,
    bool add_composed_op = false);

} // namespace jit
} // namespace torch
