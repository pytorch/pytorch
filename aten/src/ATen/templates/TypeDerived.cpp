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
#include "ATen/${DenseTensor}.h"
#include "ATen/${DenseBackend}LongTensor.h"
#include "ATen/Allocator.h"
#include "ATen/Utils.h"
#include "ATen/Half.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/THLongStorageView.h"
#include "ATen/UndefinedTensor.h"
#include "ATen/NativeFunctions.h"
#include <iostream>
#include <sstream>

#include "ATen/Config.h"
#if AT_CUDA_ENABLED()
$extra_cuda_headers
#endif

namespace at {

${Type}::${Type}(Context* context)
: Type(context, /*is_variable_or_undefined=*/false) {}
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
std::unique_ptr<Storage> ${Type}::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
    return std::unique_ptr<Storage>(
        new ${Storage}(context, size, std::move(allocator)));
}
Tensor ${Type}::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  if (retain)
    ${THTensor}_retain(${state,} (${THTensor}*) th_pointer);
  return Tensor(new ${Tensor}(context,(${THTensor}*)(th_pointer)), false);
}
std::unique_ptr<Storage> ${Type}::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  if (retain)
    ${THStorage}_retain(${state,} (${THStorage}*) th_pointer);
  return std::unique_ptr<Storage>(new ${Storage}(context, (${THStorage}*) th_pointer));
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
