#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "torch/csrc/jit/fusers/cpu/graph_fuser.h"

namespace torch { namespace jit { 

void FuseCUDAGraph(std::shared_ptr<Graph>& graph) {
  
}

void FuseCPUGraph(std::shared_ptr<Graph>& graph) {
  #if !(defined _WIN32)
    cpufuser::GraphFuser{graph->block()}.run();
  #endif // !(defined _WIN32)
}


} // namespace jit
} // namespace torch
