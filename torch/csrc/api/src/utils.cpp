#include <torch/utils.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
void manual_seed(uint64_t seed) {
  // TODO: Move this to at::Context
  at::globalContext().defaultGenerator(at::kCPU).manualSeed(seed);
  // NB: Sometimes we build with CUDA, but we don't have any GPUs
  // available. In that case, we must not seed CUDA; it will fail!
  if (at::globalContext().hasCUDA() && at::globalContext().getNumGPUs() > 0) {
    at::globalContext().defaultGenerator(at::kCUDA).manualSeedAll(seed);
  }
}
} // namespace torch
