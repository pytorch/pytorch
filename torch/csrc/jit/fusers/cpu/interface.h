#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER
#pragma once

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/cpu/fusion_compiler.h"

#include "ATen/ATen.h"

#include "torch/csrc/jit/ir.h"

#include <vector>
#include <memory>

namespace torch { namespace jit { namespace cpufuser {

inline std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group) {
  return getFusionCompiler().getFusionHandle(fusion_group);
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  return getFusionCompiler().debugLaunchGraph(graph, device, inputs);
}

} // namespace cpufuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER
