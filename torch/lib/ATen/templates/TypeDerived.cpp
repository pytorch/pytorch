#include "ATen/${Type}.h"
#include "ATen/${Storage}.h"
#include "ATen/${Tensor}.h"
#include "ATen/${Generator}.h"
#include "ATen/${Backend}ByteTensor.h"
#include "ATen/${Backend}IntTensor.h"
#include "ATen/${Backend}LongTensor.h"
#include "ATen/${SparseTensor}.h"
#include "ATen/Utils.h"
#include "ATen/THLongStorageView.h"
#include <iostream>

namespace at {

${Type}::${Type}(Context* context)
: Type(context) {}
ScalarType ${Type}::scalarType() {
  return ScalarType::${ScalarName};
}
Backend ${Type}::backend() {
  return Backend::${Backend};
}
bool ${Type}::isCuda() { return backend() == kCUDA; }
bool ${Type}::isSparse() { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool ${Type}::isDistributed() { return false; }

std::unique_ptr<Storage> ${Type}::storage() {
  return std::unique_ptr<Storage>(new ${Storage}(context));
}
std::unique_ptr<Storage> ${Type}::storage(size_t size) {
  return std::unique_ptr<Storage>(new ${Storage}(context,size));
}
std::unique_ptr<Storage> ${Type}::storageFromBlob(void * data, int64_t size) {
    return std::unique_ptr<Storage>(
      new ${Storage}(context,data,size));
}
Tensor ${Type}::unsafeTensorFromTH(void * th_pointer, bool retain) {
  if (retain)
    ${THTensor}_retain(${state,} (${THTensor}*) th_pointer);
  return Tensor(new ${Tensor}(context,(${THTensor}*)(th_pointer)), false);
}
std::unique_ptr<Generator> ${Type}::generator() {
  return std::unique_ptr<Generator>(new ${Generator}(context));
}

const char * ${Type}::toString() const {
  return ${Type}::typeString();
}
TypeID ${Type}::ID() const {
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
