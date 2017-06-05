#include "TensorLib/${Type}.h"
#include "TensorLib/${Storage}.h"

#include <iostream>

namespace tlib {

${Type}::${Type}(Context* context)
: context(context) {}
ScalarType ${Type}::scalarType() {
  return ScalarType::${ScalarName};
}
Processor ${Type}::processor() {
  return Processor::${Processor};
}
bool ${Type}::isSparse() { return false; }
bool ${Type}::isDistributed() { return false; }

Storage * ${Type}::newStorage() {
  return new ${Storage}(context);
}
Storage * ${Type}::newStorage(size_t size) {
  return new ${Storage}(context,size);
}

// example
Tensor * ${Type}::add(Tensor & a, Tensor & b) {
  std::cout << "add ${Tensor}\n";
  return &a;
}

}
