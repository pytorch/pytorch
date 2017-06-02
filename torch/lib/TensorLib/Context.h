#pragma once

#include "CPUGenerator.h"
#include "CUDAGenerator.h"
#include <memory>

namespace tlib {

class Context {
public:
  Context();
  ~Context();
  std::unique_ptr<CPUGenerator> cpu_gen;
  std::unique_ptr<CUDAGenerator> cuda_gen;
  THCState * thc_state;
};

Context * globalContext();

}
