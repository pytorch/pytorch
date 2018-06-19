#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// Erase NumberType information. This is necessary for exporting to onnx,
// and is nice to do before input shape propagation (because NumberType info
// isn't useful anymore at that step).
//
// The following things are done to erase NumberType info:
// - NumberType outputs are changed to DynamicType.
// - prim::Constant nodes' outputs get assigned their default type from ir.h
// - prim::TensorToNum, and prim::NumToTensor nodes are
//   erased.
void EraseNumberTypes(const std::shared_ptr<Graph>& graph);

}}
