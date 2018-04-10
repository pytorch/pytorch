#include "ATen/UndefinedType.h"
#include "ATen/Error.h"

namespace at {

UndefinedType::UndefinedType(Context* context)
: Type(context, /*is_variable_or_undefined=*/true) {}
ScalarType UndefinedType::scalarType() const {
  return ScalarType::Undefined;
}
Backend UndefinedType::backend() const {
  return Backend::Undefined;
}
bool UndefinedType::is_cuda() const { return false; }
bool UndefinedType::is_sparse() const { return false; }
bool UndefinedType::is_distributed() const { return false; }

std::unique_ptr<Storage> UndefinedType::storage() const {
  AT_ERROR("storage not defined for UndefinedType");
}
std::unique_ptr<Storage> UndefinedType::storage(size_t size) const {
  AT_ERROR("storage(size_t) not defined for UndefinedType");
}
std::unique_ptr<Storage> UndefinedType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  AT_ERROR("storageFromBlob not defined for UndefinedType");
}
std::unique_ptr<Storage> UndefinedType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeStorageFromTH not defined for UndefinedType");
}
std::unique_ptr<Storage> UndefinedType::storageWithAllocator(int64_t size, std::unique_ptr<Allocator> allocator) const {
  AT_ERROR("storageWithAllocator not defined for UndefinedType");
}
Tensor UndefinedType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeTensorFromTH not defined for UndefinedType");
}
std::unique_ptr<Generator> UndefinedType::generator() const {
  AT_ERROR("generator not defined for UndefinedType");
}

const char * UndefinedType::toString() const {
  return UndefinedType::typeString();
}
TypeID UndefinedType::ID() const {
  return TypeID::Undefined;
}

std::size_t UndefinedType::elementSizeInBytes() const {
  AT_ERROR("elementSizeInBytes not defined for UndefinedType");
}

Type & UndefinedType::toBackend(Backend b) const {
  if (b == Backend::Undefined) {
    return Type::toBackend(b);
  }
  AT_ERROR("toBackend not implemented for UndefinedType to non-UndefinedType");
}
Type & UndefinedType::toScalarType(ScalarType s) const {
  if (s == ScalarType::Undefined) {
    return Type::toScalarType(s);
  }
  AT_ERROR("toScalarType not implemented for UndefinedType to non-UndefinedType");
}

const char * UndefinedType::typeString() {
  return "UndefinedType";
}

Tensor & UndefinedType::s_copy_(Tensor & self, const Tensor & src, bool non_blocking) const {
  AT_ERROR("s_copy not defined for UndefinedType");
}

}
