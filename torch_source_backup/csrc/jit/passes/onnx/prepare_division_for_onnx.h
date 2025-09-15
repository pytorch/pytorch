#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit {

// Prepare division ops for ONNX export. This is necessary for and only used
// by ONNX export.
//
// The pass corrects the following:
//
// - aten::div(int, int) -> float is the python truediv operator. This doesn't
//   exist in ONNX so we cast the ints to FloatTensors
//
TORCH_API void PrepareDivisionForONNX(const std::shared_ptr<Graph>& graph);

} // namespace torch::jit
