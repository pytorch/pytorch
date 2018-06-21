#pragma once

#include "torch/csrc/jit/ir.h"

namespace torch { namespace jit {

// Erase NumberType information. This is necessary for and only used in
// exporting to ONNX.
//
// The following things are done to erase NumberType info:
// - NumberType outputs are changed to DynamicType.
// - Any aten::type_as nodes that are added to correct Number math
//   are removed because ONNX export does not support them.
// - prim::Constant nodes' outputs get assigned their default type from ir.h
// - prim::TensorToNum, and prim::NumToTensor nodes are erased.
//
// The pass assumes that DCE will be called sometime after.
void EraseNumberTypes(const std::shared_ptr<Graph>& graph);

}}
