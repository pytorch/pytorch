#pragma once

#include "ATen/ATen.h"

void manual_seed(uint64_t seed, at::DeviceType backend) {
  if (backend == at::kCPU) {
    auto& cpu_gen = at::globalContext().getDefaultGenerator(at::kCPU);
    cpu_gen.setCurrentSeed(seed);
  } else if (backend == at::kCUDA && at::hasCUDA()) {
    auto& cuda_gen = at::globalContext().getDefaultGenerator(at::kCUDA);
    cuda_gen.setCurrentSeed(seed);
  }
}
