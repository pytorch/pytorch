#include <torch/utils.h>

#include <ATen/Context.h>

#include <cstddef>

namespace torch {
void manual_seed(uint64_t seed) {
  // TODO: Move this to at::Context
  at::globalContext().defaultGenerator(at::Backend::CPU).manualSeed(seed);
  if (at::globalContext().hasCUDA()) {
    at::globalContext().defaultGenerator(at::Backend::CUDA).manualSeedAll(seed);
  }
}
} // namespace torch
