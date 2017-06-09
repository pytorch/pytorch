#include "TensorLib/${Type}.h"
#include "TensorLib/${Storage}.h"
#include "TensorLib/${Tensor}.h"
#include "TensorLib/${Processor}Generator.h"
#include "TensorLib/${Processor}ByteTensor.h"
#include "TensorLib/${Processor}IntTensor.h"
#include "TensorLib/${Processor}LongTensor.h"
#include "TensorLib/Utils.h"
#include "TensorLib/THStorageView.h"
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

std::unique_ptr<Storage> ${Type}::newStorage() {
  return std::unique_ptr<Storage>(new ${Storage}(context));
}
std::unique_ptr<Storage> ${Type}::newStorage(size_t size) {
  return std::unique_ptr<Storage>(new ${Storage}(context,size));
}
std::unique_ptr<Generator> ${Type}::newGenerator() {
  return std::unique_ptr<Generator>(new ${Processor}Generator(context));
}

const char * ${Type}::toString() const {
  return ${Type}::typeString();
}
int ${Type}::ID() const {
  return ${TypeID};
}

const char * ${Type}::typeString() {
  return "${Type}";
}

/* example
Tensor * ${Type}::add(Tensor & a, Tensor & b) {
  std::cout << "add ${Tensor}\n";
  return &a;
}
*/

${type_derived_method_definitions}

}
