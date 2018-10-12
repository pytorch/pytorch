#include "torch/csrc/jit/fusers/interface.h"

#include "torch/csrc/jit/fusers/Config.h"

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fusers/cpu/interface.h"
#endif // USE_CPU_FUSER

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fusers/cuda/interface.h"
#endif // USE_CUDA_FUSER

#include <stdexcept>

namespace torch { namespace jit {

namespace detail {

bool cpu_fuser_enabled = false;

} // namespace detail

// Pure virtual destructor definition
FusionHandle::~FusionHandle() { }

std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group) {
  const auto device = fusion_group->i(attr::device);
  if (device == kCPUDevice) {
    #if USE_CPU_FUSER
      return cpufuser::getFusionHandle(fusion_group);
    #endif
    throw std::runtime_error("CPU fusion is not supported on this build.");
  }

  #if USE_CUDA_FUSER
    return cudafuser::getFusionHandle(fusion_group);
  #endif // USE_CUDA_FUSER

  throw std::runtime_error("CUDA fusion is not supported on this build.");
}

bool canFuseOnCPU() {
  #if USE_CPU_FUSER
    return detail::cpu_fuser_enabled;
  #endif // USE_CPU_FUSER

  return false;
}

bool canFuseOnGPU() {
  #if USE_CUDA_FUSER
    return true;
  #endif  // USE_CUDA_FUSER

  return false;
}

void overrideCanFuseOnCPU(bool value) {
  detail::cpu_fuser_enabled = value;
}

std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs) {
  if (device == kCPUDevice) {
    #if USE_CPU_FUSER
      return cpufuser::debugLaunchGraph(graph, device, inputs);
    #endif // USE_CPU_FUSER
    throw std::runtime_error("CPU fusion is not supported on this build.");
  }

  #if USE_CUDA_FUSER
    return cudafuser::debugLaunchGraph(graph, device, inputs);
  #endif // USE_CUDA_FUSER

  throw std::runtime_error("CUDA fusion is not supported on this build.");
}

} // namespace jit
} // namespace torch
