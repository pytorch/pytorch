#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// Prepare binary ops for LLGA
//
// The pass does the following:
//
// - Convert scalar input of aten::add and aten::mul into Float tensor with
//   dimension [1]
//
// - Decompose fused add into aten::mul + aten::add when alpha != 1.0
//
// - Eliminate identity add/mul, i.e., tensor + 0, tensor * 1
//
void PrepareBinaryForLLGA(const std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
