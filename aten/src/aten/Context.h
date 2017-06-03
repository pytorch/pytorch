#pragma once

#include "CPUGenerator.h"
#include "CUDAGenerator.h"
#include <memory>
#include "Type.h"

namespace tlib {

class Context {
public:
  Context();
  Type & getType(Processor p, ScalarType s) {
    return *type_registry[static_cast<int>(p)][static_cast<int>(s)];
  }
  ~Context();
  std::unique_ptr<CPUGenerator> cpu_gen;
  std::unique_ptr<CUDAGenerator> cuda_gen;
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Processor::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  THCState * thc_state;
};

Context * globalContext();

}
