#include <ATen/Config.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "torch/detail.h"

namespace torch {
void setSeed(uint64_t seed) {
  // TODO: Move this to at::Context
  at::globalContext().defaultGenerator(at::Backend::CPU).manualSeed(seed);
  if (at::globalContext().hasCUDA()) {
    at::globalContext().defaultGenerator(at::Backend::CUDA).manualSeedAll(seed);
  }
}

int getNumGPUs() {
  return at::globalContext().getNumGPUs();
}

bool hasCuda() {
  // NB: the semantics of this are different from at::globalContext().hasCUDA();
  // ATen's function tells you if you have a working driver and CUDA build,
  // whereas this function also tells you if you actually have any GPUs.
  return getNumGPUs() > 0;
}

bool hasCudnn() {
  return hasCuda() && at::globalContext().hasCuDNN();
}

} // namespace torch
