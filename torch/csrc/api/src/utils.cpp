#include <torch/utils.h>
#include <torch/cuda.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
void manual_seed(uint64_t seed) {
  // TODO: Move this to at::Context
  at::globalContext().defaultGenerator(at::kCPU).manualSeed(seed);
  if (cuda::is_available()) {  // built with CUDA, have at least one device
    at::globalContext().defaultGenerator(at::kCUDA).manualSeedAll(seed);
  }
}
} // namespace torch
