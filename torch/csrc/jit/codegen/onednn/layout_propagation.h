#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

bool PropagateLayoutEnabled();
TORCH_API bool setPropagateLayoutMode(bool mode);
void PropagateLayout(const std::shared_ptr<Graph>& graph);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
