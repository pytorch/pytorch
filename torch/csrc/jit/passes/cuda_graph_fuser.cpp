#include <torch/csrc/jit/passes/cuda_graph_fuser.h>

namespace torch {
namespace jit {

bool getIsNVFuserEnabled() {
  return RegisterCudaFuseGraph::isRegistered();
}

}
}
