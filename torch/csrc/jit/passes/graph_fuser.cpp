#include "torch/csrc/jit/passes/graph_fuser.h"

#include "torch/csrc/jit/fusers/cpu/cpu_fusion_pass.h"
#include "torch/csrc/jit/fusers/cuda/cuda_fusion_pass.h"

#include "torch/csrc/jit/fusers/cpu/cpu_fuser_interface.h"

namespace torch { namespace jit {

void FuseGraph(std::shared_ptr<Graph>& graph) {
  if (canCompileOnCPU()) FuseCPUGraph(graph);

  #if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
    FuseCUDAGraph(graph);
  #endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
}

} // namespace jit
} // namespace torch

