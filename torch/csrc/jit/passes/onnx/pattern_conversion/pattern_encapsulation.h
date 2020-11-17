#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Introduction
//
// The encapsulation part will find the nodes of patterns, like how other pre-onnx passes are written. But instead of converting the nodes, it will encapsulate them into a sub-block of a new placeholder node. This part is called before onnx pass, so it runs before calling symbolic functions.
//
// Note: Why separate the pass into two parts
//
// The purpose of this is to support conversions that depend on shape and type information. Shape and type information is only available after _jit_pass_onnx, which converts aten nodes to onnx nodes. So there is a mutual dependency issue here. _jit_pass_onnx depends on preprocess passes to convert aten nodes into convertable condition, and preprocess passes depends on _jit_pass_onnx to convert upstream nodes and apply onnx shape inference.
//
// Note: Edit Pattern Encapsulation
//
TORCH_API void EncapsulatePatternIntoSubblock(
    const std::shared_ptr<Graph>& graph);

} // namespace jit
} // namespace torch
