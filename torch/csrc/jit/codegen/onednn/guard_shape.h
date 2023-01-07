#pragma once

#include <torch/csrc/jit/ir/ir.h>

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

void prepareFusionGroupAndGuardOutputs(Block* block);

} // namespace onednn
} // namespace fuser
} // namespace jit
} // namespace torch
