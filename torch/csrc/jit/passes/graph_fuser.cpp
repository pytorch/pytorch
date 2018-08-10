#include "torch/csrc/jit/passes/graph_fuser.h"

#include "torch/csrc/jit/fusers/cpu/cpu_fusion_pass.h"
#include "torch/csrc/jit/fusers/cuda/cuda_fusion_pass.h"

namespace torch { namespace jit {

void FuseGraph(std::shared_ptr<Graph>& graph) {
  FuseCPUGraph(graph);
  #ifdef USE_CUDA
    FuseCUDAGraph(graph);
  #endif // USE_CUDA
}

} // namespace jit
} // namespace torch
