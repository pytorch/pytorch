// required for old g++ to compile PRId64 macros, see
// https://github.com/pytorch/pytorch/issues/3571
// for context
#define __STDC_FORMAT_MACROS

#include "ATen/${Type}.h"

// ${generated_comment}

#include "ATen/${Generator}.h"
#include "ATen/Allocator.h"
#include "ATen/DeviceGuard.h"
#include "ATen/NativeFunctions.h"
#include "ATen/UndefinedTensor.h"
#include "ATen/Utils.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/core/Half.h"
#include "ATen/core/optional.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "ATen/Config.h"
$extra_cuda_headers

namespace at {

${Type}::${Type}(Context* context)
  : Type(context, ${Backend}TensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}
ScalarType ${Type}::scalarType() const {
  return ScalarType::${ScalarName};
}
Backend ${Type}::backend() const {
  return Backend::${Backend};
}
bool ${Type}::is_cuda() const { return backend() == Backend::CUDA || backend() == Backend::SparseCUDA; }
bool ${Type}::is_sparse() const { return backend() == Backend::SparseCPU || backend() == Backend::SparseCUDA; }
bool ${Type}::is_distributed() const { return false; }

std::unique_ptr<Storage> ${Type}::storage(bool resizable) const {
  AT_ERROR("storage not supported on sparse");
}
std::unique_ptr<Storage> ${Type}::storage(size_t size, bool resizable) const {
  AT_ERROR("storage not supported on sparse");
}
std::unique_ptr<Storage> ${Type}::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  AT_ERROR("storage not supported on sparse");
}
std::unique_ptr<Storage> ${Type}::storageWithAllocator(int64_t size, Allocator* allocator) const {
  AT_ERROR("storage not supported on sparse");
}
Tensor ${Type}::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeTensorFromTH not supported on sparse");
}
std::unique_ptr<Storage> ${Type}::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeTensorFromTH not supported on sparse");
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

size_t ${Type}::elementSizeInBytes() const {
  return sizeof(${ScalarType});
}

const char * ${Type}::typeString() {
  return "${Type}";
}

${type_derived_method_definitions}

}
