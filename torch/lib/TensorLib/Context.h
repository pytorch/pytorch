#pragma once

#include <memory>
#include "TensorLib/Generator.h"
#include "TensorLib/Type.h"

class THCState;

namespace tlib {

class Context {
public:
  Context();
  Type & getType(Backend p, ScalarType s) {
    auto & type = type_registry[static_cast<int>(p)][static_cast<int>(s)];
    if(!type)
      throw std::runtime_error("Type %s is not enabled (TODO encode type as string)");
    return *type;
  }
  Generator & defaultGenerator(Backend p) {
    auto & generator = generator_registry[static_cast<int>(p)];
    if(!generator)
      throw std::runtime_error("backend type not enabled (TODO encode name as string)");
    return *generator;
  }
  Type & defaultType() {
    return *current_default_type;
  }
  bool hasCUDA() const;
  void setDefaultType(Type & t) {
    current_default_type = &t;
  }
  ~Context();
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(Backend::NumOptions)];
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Backend::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  THCState * thc_state;
  Type * current_default_type;
};

Context & globalContext();


static inline Type& getType(Backend p, ScalarType s) {
  return globalContext().getType(p,s);
}

static inline Type& CPU(ScalarType s) {
  return getType(Backend::CPU, s);
}

static inline Type& CUDA(ScalarType s) {
  return getType(Backend::CUDA, s);
}


}
