#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch::jit::fuser::onednn {

void prepareFusionGroupAndGuardOutputs(Block* block);

} // namespace torch::jit::fuser::onednn
