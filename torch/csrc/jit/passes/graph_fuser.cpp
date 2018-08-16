#include "torch/csrc/jit/passes/graph_fuser.h"
#include "torch/csrc/jit/fusers/fuser_interface.h"

namespace torch { namespace jit {

void FuseGraph(std::shared_ptr<Graph>& graph) {
  FuseCPUGraph(graph);
  FuseCUDAGraph(graph);
}

} // namespace jit
} // namespace torch
