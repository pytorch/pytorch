#pragma once

#include "lazy_tensor_core/csrc/compiler/node_lowering.h"

namespace torch_lazy_tensors {
namespace compiler {

NodeLowering* GetTSNodeLowering();
std::unique_ptr<NodeLowering> CreateTSNodeLowering(ir::LoweringContext* loctx);

}  // namespace compiler
}  // namespace torch_lazy_tensors
