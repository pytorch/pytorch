#pragma once
#include "ATen/CPUGenerator.h"

namespace at {
static inline CPUGenerator * check_generator(Generator* expr) {
  if(auto result = dynamic_cast<CPUGenerator*>(expr))
    return result;
  runtime_error("Expected a 'CPUGenerator' but found 'CUDAGenerator'");
}
}
