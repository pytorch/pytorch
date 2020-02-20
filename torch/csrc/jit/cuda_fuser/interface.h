#pragma once
#include <atomic>
#include <torch/csrc/WindowsTorchApiMacro.h>
namespace torch {
namespace jit{
TORCH_API std::atomic<bool>& getCudaFusionGroupOptimizationPassMode();
TORCH_API std::atomic<bool>& getCudaFusionGroupExecutorMode();
}
}
