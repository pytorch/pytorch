#pragma once

#include <memory>
#include "TensorLib/Generator.h"
#include "TensorLib/Type.h"

class THCState;

namespace tlib {

class Context {
public:
  Context();
  Type & getType(Processor p, ScalarType s) {
    auto & type = type_registry[static_cast<int>(p)][static_cast<int>(s)];
    if(!type)
      throw std::runtime_error("type is not enabled (TODO encode type as string)");
    return *type;
  }
  Generator & defaultGenerator(Processor p) {
    auto & generator = generator_registry[static_cast<int>(p)];
    if(!generator)
      throw std::runtime_error("processor type not enabled (TODO encode name as string)");
    return *generator;
  }
  Type & defaultType() {
    return *current_default_type;
  }
  void setDefaultType(Type & t) {
    current_default_type = &t;
  }
  ~Context();
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(Processor::NumOptions)];
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Processor::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  THCState * thc_state;
  Type * current_default_type;
};

Context * globalContext();


static inline Type& getType(Processor p, ScalarType s) {
  return globalContext()->getType(p,s);
}

static inline Type& CPU(ScalarType s) {
  return getType(Processor::CPU, s);
}

static inline Type& CUDA(ScalarType s) {
  return getType(Processor::CUDA, s);
}


}
