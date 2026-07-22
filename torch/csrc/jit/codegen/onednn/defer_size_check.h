#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit::fuser::onednn {

void DeferSizeCheck(std::shared_ptr<Graph>& graph);

} // namespace torch::jit::fuser::onednn
