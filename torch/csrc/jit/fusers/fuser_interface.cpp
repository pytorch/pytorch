#include "torch/csrc/jit/fusers/fuser_interface.h"

#if !(defined _WIN32)
  #include "torch/csrc/jit/fusers/cpu/fusion_compiler.h"
  #include "torch/csrc/jit/fusers/cpu/graph_fuser.h"
#endif // !(defined _WIN32)

namespace torch { namespace jit { 

void FuseCUDAGraph(std::shared_ptr<Graph>& graph) {
  
}

void FuseCPUGraph(std::shared_ptr<Graph>& graph) {
  #if !(defined _WIN32)
    if (cpufuser::getCompiler().canCompileOnCPU())
      cpufuser::GraphFuser{graph->block()}.run();
  #endif // !(defined _WIN32)
}


} // namespace jit
} // namespace torch
