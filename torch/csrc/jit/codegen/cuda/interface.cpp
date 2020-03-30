#include <torch/csrc/jit/codegen/cuda/interface.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

CudaFuserInterface* getFuserInterface() {
  static CudaFuserInterface fuser_interface_;
  return &fuser_interface_;
}

void compileFusionGroup(Node* fusion_node) {
  TORCH_CHECK(
      getFuserInterface()->fn_compile_n_ != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_compile_n_(fusion_node);
}

void runFusionGroup(const Node* fusion_node, Stack& stack) {
  TORCH_CHECK(
      getFuserInterface()->fn_run_n_s_ != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_run_n_s_(fusion_node, stack);
}

void fuseGraph(std::shared_ptr<Graph>& graph) {
  TORCH_CHECK(
      getFuserInterface()->fn_fuse_graph != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_fuse_graph(graph);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
