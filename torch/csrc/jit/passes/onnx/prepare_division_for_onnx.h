#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// Prepare division ops for ONNX export. This is necessary for and only used
// by ONNX export.
//
// The pass corrects the following:
//
// - aten::div(int, int) -> float is the python truediv operator. This doesn't
//   exist in ONNX so we express it in terms of ONNX ops.
//
TORCH_API void PrepareDivisionForONNX(const std::shared_ptr<Graph>& graph);

}}
