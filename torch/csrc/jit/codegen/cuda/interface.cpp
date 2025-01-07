#include <torch/csrc/jit/codegen/cuda/interface.h>

#include <ATen/DynamicLibrary.h>
#include <ATen/core/dispatch/OperatorOptions.h>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/TensorShape.h>
#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/register_ops_utils.h>

namespace torch::jit::fuser::cuda {

static std::atomic<bool> cuda_fusion_guard_mode{true};

bool isEnabled() {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::isEnabled() is deprecated");
  return false;
}

bool setEnabled(bool is_enabled) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::setEnabled() is deprecated");
  TORCH_INTERNAL_ASSERT(
      !is_enabled,
      "nvfuser support in torchscript is removed and cannot be enabled!");
  return false;
}

bool canBeEnabled() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::nvfuserCanBeEnabled() is deprecated");
  return false;
}

bool getSingletonFusion() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::getSingletonFusion() is deprecated");
  return false;
}

bool setSingletonFusion(bool value) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::setSingletonFusion() is deprecated");
  TORCH_INTERNAL_ASSERT(
      !value,
      "nvfuser support in torchscript is removed and singleton fusion cannot be enabled!");
  return false;
}

bool getHorizontalFusion() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::getHorizontalFusion() is deprecated");
  return false;
}

bool setHorizontalFusion(bool value) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::setHorizontalFusion() is deprecated");
  TORCH_INTERNAL_ASSERT(
      !value,
      "nvfuser support in torchscript is removed and horizontal fusion cannot be enabled!");
  return false;
}

std::atomic<bool>& getCudaFusionGuardMode() {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::getCudaFusionGuardMode() is deprecated");
  return cuda_fusion_guard_mode;
}

CudaFuserInterface* getFuserInterface() {
  static CudaFuserInterface fuser_interface_;
  return &fuser_interface_;
}

void compileFusionGroup(Node* fusion_node) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::compileFusionGroup() is deprecated");
  TORCH_CHECK(
      getFuserInterface()->fn_compile_n != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_compile_n(fusion_node);
}

void runFusionGroup(const Node* fusion_node, Stack& stack) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::runFusionGroup() is deprecated");
  TORCH_CHECK(
      getFuserInterface()->fn_run_n_s != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_run_n_s(fusion_node, stack);
}

void fuseGraph(std::shared_ptr<Graph>& graph) {
  if (!isEnabled()) {
    return;
  }

  TORCH_WARN_ONCE("nvfuser integration in TorchScript is deprecated.");
  TORCH_CHECK(
      getFuserInterface()->fn_fuse_graph != nullptr,
      "Running the CUDA fuser requires a CUDA build.");
  getFuserInterface()->fn_fuse_graph(graph);
}

bool canFuseNode(const Node* node) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::canFuseNode() is deprecated");
  return getFuserInterface()->fn_can_fuse_n != nullptr &&
      getFuserInterface()->fn_can_fuse_n(node);
}

void InsertProfileNodesForCUDAFuser(ProfilingRecord* pr) {
  TORCH_WARN_ONCE(
      "torch::jit::fuser::cuda::InsertProfileNodesForCUDAFuser() is deprecated");
  if (getFuserInterface()->fn_insert_profile_inodes) {
    getFuserInterface()->fn_insert_profile_inodes(pr);
  }
}

bool profileNode(const Node* node) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::profileNode() is deprecated");
  return getFuserInterface()->fn_profile_n != nullptr &&
      getFuserInterface()->fn_profile_n(node);
}

bool skipNode(const std::string& symbol_str, bool flip) {
  TORCH_WARN_ONCE("torch::jit::fuser::cuda::skipNode() is deprecated");
  return getFuserInterface()->fn_skip_n != nullptr &&
      getFuserInterface()->fn_skip_n(symbol_str, flip);
}

} // namespace torch::jit::fuser::cuda
