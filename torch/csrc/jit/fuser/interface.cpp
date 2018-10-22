#include "torch/csrc/jit/fuser/interface.h"

#include "torch/csrc/jit/fuser/config.h"
#include "torch/csrc/jit/fuser/compiler.h"
#include "torch/csrc/jit/fuser/executor.h"
#include "torch/csrc/jit/fuser/fallback.h"

#if USE_CPU_FUSER
  #include "torch/csrc/jit/fuser/cpu/interface.h"
#endif // USE_CPU_FUSER

#if USE_CUDA_FUSER
  #include "torch/csrc/jit/fuser/cuda/interface.h"
#endif // USE_CUDA_FUSER

#include <stdexcept>

namespace torch { namespace jit {

namespace detail {

bool cpu_fuser_enabled = false;

} // namespace detail

// Pure virtual destructor definition
FusionHandle::~FusionHandle() { }

void registerFusion(int64_t& key, const Node* fusion_group) {
  fuser::registerFusion(key, fusion_group); 
}

void runFusion(const int64_t key, Stack& stack) {
  try {
    fuser::runFusion(key, stack);
  } catch (...) {
    fuser::runFallback(key, stack);
  }
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
      return fuser::cpu::debugLaunchGraph(graph, device, inputs);
    #endif // USE_CPU_FUSER
    throw std::runtime_error("CPU fusion is not supported on this build.");
  }

  #if USE_CUDA_FUSER
    return fuser::cuda::debugLaunchGraph(graph, device, inputs);
  #endif // USE_CUDA_FUSER

  throw std::runtime_error("CUDA fusion is not supported on this build.");
}

} // namespace jit
} // namespace torch
