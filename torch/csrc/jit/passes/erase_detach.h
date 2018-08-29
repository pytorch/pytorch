#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// Erase detach. This is necessary for export to ONNX, since ONNX
// does not support training. This is safe because, again,
// ONNX does not suppor ttracing.
TORCH_API void EraseDetach(const std::shared_ptr<Graph>& graph);

}}
