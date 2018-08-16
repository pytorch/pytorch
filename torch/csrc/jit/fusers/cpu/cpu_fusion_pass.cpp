#include "torch/csrc/jit/fusers/fuser_interface.h"

namespace torch { namespace jit {

void FuseCPUGraph(std::shared_ptr<Graph>& graph) {
  #if !(defined _WIN32)
    // if (canCompileOnCPU()) GraphFuser(graph->block()).run();
  #endif // !(defined _WIN32)
}


} // namespace jit
} // namespace torch


