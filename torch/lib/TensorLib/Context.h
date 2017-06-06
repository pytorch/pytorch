#pragma once

#include <memory>
#include "TensorLib/Generator.h"
#include "TensorLib/Type.h"

namespace tlib {

class THCState;

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
  ~Context();
  std::unique_ptr<Generator>
    generator_registry[static_cast<int>(Processor::NumOptions)];
  std::unique_ptr<Type> type_registry
    [static_cast<int>(Processor::NumOptions)]
    [static_cast<int>(ScalarType::NumOptions)];
  THCState * thc_state;
};

Context * globalContext();

}
