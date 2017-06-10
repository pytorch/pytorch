#include "TensorLib/${Type}.h"
#include "TensorLib/${Storage}.h"
#include "TensorLib/${Tensor}.h"
#include "TensorLib/${Backend}Generator.h"
#include "TensorLib/${Backend}ByteTensor.h"
#include "TensorLib/${Backend}IntTensor.h"
#include "TensorLib/${Backend}LongTensor.h"
#include "TensorLib/Utils.h"
#include "TensorLib/THLongStorageView.h"
#include <iostream>

namespace tlib {

${Type}::${Type}(Context* context)
: context(context) {}
ScalarType ${Type}::scalarType() {
  return ScalarType::${ScalarName};
}
Backend ${Type}::backend() {
  return Backend::${Backend};
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
  return std::unique_ptr<Generator>(new ${Backend}Generator(context));
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
