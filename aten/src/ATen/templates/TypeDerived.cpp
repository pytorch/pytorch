// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/${Type}.h"
#include "ATen/${Storage}.h"
#include "ATen/${Tensor}.h"
#include "ATen/${Generator}.h"
#include "ATen/${Backend}ByteTensor.h"
#include "ATen/${Backend}IntTensor.h"
#include "ATen/${Backend}LongTensor.h"
#include "ATen/${SparseTensor}.h"
#include "ATen/Utils.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/THLongStorageView.h"
#include "ATen/UndefinedTensor.h"
#include "ATen/NativeFunctions.h"
#include <iostream>
#include <sstream>

namespace at {

${Type}::${Type}(Context* context)
: Type(context) {}
ScalarType ${Type}::scalarType() const {
  return ScalarType::${ScalarName};
}
Backend ${Type}::backend() const {
  return Backend::${Backend};
}
bool ${Type}::is_cuda() const { return backend() == kCUDA || backend() == kSparseCUDA; }
bool ${Type}::is_sparse() const { return backend() == kSparseCPU || backend() == kSparseCUDA; }
bool ${Type}::is_distributed() const { return false; }

std::unique_ptr<Storage> ${Type}::storage() const {
  return std::unique_ptr<Storage>(new ${Storage}(context));
}
std::unique_ptr<Storage> ${Type}::storage(size_t size) const {
  return std::unique_ptr<Storage>(new ${Storage}(context,size));
}
std::unique_ptr<Storage> ${Type}::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
    return std::unique_ptr<Storage>(
      new ${Storage}(context,data,size,deleter));
}
Tensor ${Type}::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    ${THTensor}_retain(${state,} (${THTensor}*) th_pointer);
  return Tensor(new ${Tensor}(context,(${THTensor}*)(th_pointer)), false);
}
std::unique_ptr<Generator> ${Type}::generator() const {
  return std::unique_ptr<Generator>(new ${Generator}(context));
}

const char * ${Type}::toString() const {
  return ${Type}::typeString();
}
TypeID ${Type}::ID() const {
  return ${TypeID};
}

std::size_t ${Type}::elementSizeInBytes() const {
  return sizeof(${ScalarType});
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
