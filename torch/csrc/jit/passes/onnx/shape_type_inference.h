#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {

// Utilize ONNX Shape Inference for node.
// The node must have ONNX namespace, and is valid ONNX node accroding to spec.
// On successful ONNX shape inference runs, the function updates output types of
// n with inferred shape and type. Otherwise n is unchanged.
TORCH_API void ONNXShapeTypeInference(Node* n, int opset_version);

} // namespace jit
} // namespace torch
