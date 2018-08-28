#pragma once

#include "torch/csrc/jit/fusers/fusion_interface.h"
#include "torch/csrc/jit/fusers/cuda/cuda_fusion_compiler.h"

#include "torch/csrc/jit/ir.h"

#include "ATen/ATen.h"

#include <vector>
#include <memory>

namespace torch { namespace jit { namespace cudafuser {

inline std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group) {
  return getFusionCompiler().getFusionHandle(fusion_group);
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  return getFusionCompiler().debugLaunchGraph(graph, device, inputs);
}

} // namespace cudafuser
} // namespace jit 
} // namespace torch
