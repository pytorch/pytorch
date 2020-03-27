#include <torch/csrc/jit/codegen/cuda/interface.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

CudaFuserInterface* getFuserInterface() {
  static CudaFuserInterface fuser_interface_;
  return &fuser_interface_;
}

bool isFusable(const Node* node) {
  TORCH_CHECK(getFuserInterface()->fn_is_fusible_n_ != nullptr, "fn_is_fusible_n_ not initialized");
  return getFuserInterface()->fn_is_fusible_n_(node);
}

bool isFusable(const Node* fusion, const Node* node) {
  TORCH_CHECK(getFuserInterface()->fn_is_fusible_n_n_ != nullptr, "fn_is_fusible_n_n_ not initialized");
  return getFuserInterface()->fn_is_fusible_n_n_(fusion, node);
}

void compileFusionGroup(Node* fusion_node) {
  TORCH_CHECK(getFuserInterface()->fn_compile_n_ != nullptr, "fn_compile_n_ not initialized");
  getFuserInterface()->fn_compile_n_(fusion_node);
}

void runFusionGroup(const Node* fusion_node, Stack& stack) {
  TORCH_CHECK(getFuserInterface()->fn_run_n_s_ != nullptr, "fn_run_n_s_ not initialized");
  getFuserInterface()->fn_run_n_s_(fusion_node, stack);
}

}}}} // namespace torch::jit::fuser::cuda
