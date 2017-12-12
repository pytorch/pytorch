#include "ATen/UndefinedType.h"

namespace at {

UndefinedType::UndefinedType(Context* context)
: Type(context) {}
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
  runtime_error("storage not defined for UndefinedType");
}
std::unique_ptr<Storage> UndefinedType::storage(size_t size) const {
  runtime_error("storage(size_t) not defined for UndefinedType");
}
std::unique_ptr<Storage> UndefinedType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  runtime_error("storageFromBlob not defined for UndefinedType");
}
std::unique_ptr<Storage> UndefinedType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  runtime_error("unsafeStorageFromTH not defined for UndefinedType");
}
Tensor UndefinedType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  runtime_error("unsafeTensorFromTH not defined for UndefinedType");
}
std::unique_ptr<Generator> UndefinedType::generator() const {
  runtime_error("generator not defined for UndefinedType");
}

const char * UndefinedType::toString() const {
  return UndefinedType::typeString();
}
TypeID UndefinedType::ID() const {
  return TypeID::Undefined;
}

std::size_t UndefinedType::elementSizeInBytes() const {
  runtime_error("elementSizeInBytes not defined for UndefinedType");
}

Type & UndefinedType::toBackend(Backend b) const {
  if (b == Backend::Undefined) {
    return Type::toBackend(b);
  }
  runtime_error("toBackend not implemented for UndefinedType to non-UndefinedType");
}
Type & UndefinedType::toScalarType(ScalarType s) const {
  if (s == ScalarType::Undefined) {
    return Type::toScalarType(s);
  }
  runtime_error("toScalarType not implemented for UndefinedType to non-UndefinedType");
}

const char * UndefinedType::typeString() {
  return "UndefinedType";
}

Tensor & UndefinedType::s_copy_(Tensor & self, const Tensor & src, bool async) const {
  runtime_error("s_copy not defined for UndefinedType");
}

}
