#include <torch/utils.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
void manual_seed(uint64_t seed) {
  // TODO: Move this to at::Context
  at::globalContext().getDefaultGenerator(at::kCPU).setCurrentSeed(seed);
  // NB: Sometimes we build with CUDA, but we don't have any GPUs
  // available. In that case, we must not seed CUDA; it will fail!
  if (at::globalContext().hasCUDA() && at::globalContext().getNumGPUs() > 0) {
    int64_t num_gpus = at::globalContext().getNumGPUs();
    for (auto i = decltype(num_gpus){0}; i < num_gpus; ++i) {
      at::globalContext().getDefaultGenerator(at::kCUDA, i).setCurrentSeed(seed);
    }
  }
}
} // namespace torch
