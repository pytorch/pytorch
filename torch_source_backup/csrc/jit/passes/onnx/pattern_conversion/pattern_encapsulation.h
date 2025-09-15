#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Introduction
//
// The encapsulation part will find the nodes of patterns, like how other
// pre-onnx passes are written. But instead of converting the nodes, it will
// encapsulate them into a sub-block of a new placeholder node. This part is
// called before onnx pass, so it runs before calling symbolic functions.
//
// Note: Why separate the function into two parts
//
// The purpose is to support conversions that depend on shape and type
// information. Shape and type information is only available after
// _jit_pass_onnx, which converts aten nodes to onnx nodes. So there is a
// interdependent issue. _jit_pass_onnx depends on preprocess passes to convert
// aten nodes into convertable condition, and preprocess passes depend on
// _jit_pass_onnx to convert upstream nodes and apply onnx shape inference.
// Separating the pass into two parts breaks the interdependency.
//
// Note: Edit Pattern Encapsulation
//
// Encapsulation step identifies the pattern, and copies the nodes into
// the subblock of a new placeholder node. The outputs of the new placeholder
// node are used in place of the original nodes instead. The category of the
// pattern is stored as attr::name.
TORCH_API std::optional<Node*> EncapsulatePatternIntoSubblock(Node* n);

} // namespace torch::jit
