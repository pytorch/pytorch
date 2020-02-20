#include <torch/csrc/jit/cuda_fuser/interface.h>

namespace torch {
namespace jit{

static std::atomic<bool> kCudaFusionGroupOptimizationPassMode{false};
static std::atomic<bool> kCudaFusionGroupExecutorMode{false};

std::atomic<bool>& getCudaFusionGroupOptimizationPassMode() {
  return kCudaFusionGroupOptimizationPassMode;
}

std::atomic<bool>& getCudaFusionGroupExecutorMode() {
  return kCudaFusionGroupExecutorMode;
}

}
}
