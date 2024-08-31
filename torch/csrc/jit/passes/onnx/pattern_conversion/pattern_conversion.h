#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/utils/pybind.h>

namespace torch {
namespace jit {

// Introduction
//
// The conversion part is called inside the onnx pass.
// In onnx pass, _run_symbolic_function will be called for each node in
// topological order. When it reaches the placeholder node, this function will
// be invoked. It will convert the nodes inside the sub-block based on pattern.
// By that time, it will have shape/type of upstream operators available. After
// the conversion is complete, the placeholder node will be removed, and nodes
// inside its sub-block converted. NodeToONNX will be called for these
// nodes, and they will be converted from ATen operator to ONNX operator.
//
// Note: Edit Pattern Conversion
//
// Each pattern is differentiated by the name attribute of placeholder node.
// The placeholder node is part of torch IR graph, After this function, the aten
// nodes under placeholder node subblock will be converted to ONNX and appended
// to the new_block, which is under the new ONNX graph. For the pattern
// conversion code, it can be divided into three parts.
//      1. Nodes in this pattern should be captured inside the subblock of
//         Placeholder node after pattern encapsulation[see
//         pattern_encapsulation.h]. These nodes will be converted based on
//         pattern. This part of conversion is from aten to aten. It happens on
//         the torch IR graph inside placeholder node subblock.
//      2. The second part of conversion is to convert the aten nodes produced
//         into ONNX. This is done by calling NodeToONNX for each node. The new
//         ONNX nodes are appended to the new_block, which is under the new ONNX
//         graph.
//      3. The last part of conversion is to find and return, in the same order,
//         the ONNX outputs corresponding to the original output for the
//         placeholder node.
TORCH_API std::vector<Value*> ConvertPatternFromSubblock(
    Block* new_block,
    Node* old_node,
    py::dict& env,
    py::set& values_in_env);

} // namespace jit
} // namespace torch
