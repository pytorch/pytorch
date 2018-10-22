#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"
#include "torch/csrc/jit/fuser/cuda/fusion_compiler.h"

#include <vector>
#include <memory>

namespace torch { namespace jit { namespace fuser { namespace cuda {

inline std::shared_ptr<FusionHandle> getFusionHandle(
  const KernelSpec& spec
, const int device) {
  return getFusionCompiler().getFusionHandle(spec, device);
}

inline std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  return getFusionCompiler().debugLaunchGraph(graph, device, inputs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CUDA_FUSER
