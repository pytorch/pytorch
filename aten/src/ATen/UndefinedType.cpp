#include <ATen/UndefinedType.h>
#include <c10/util/Exception.h>

namespace at {

UndefinedType::UndefinedType()
    : TypeDefault(UndefinedTensorId(), /*is_variable=*/false, /*is_undefined=*/true) {}
ScalarType UndefinedType::scalarType() const {
  return ScalarType::Undefined;
}
caffe2::TypeMeta UndefinedType::typeMeta() const {
  return scalarTypeToTypeMeta(scalarType());
}
Backend UndefinedType::backend() const {
  return Backend::Undefined;
}

Allocator* UndefinedType::allocator() const {
  AT_ERROR("allocator not defined for UndefinedType");
}

Device UndefinedType::getDeviceFromPtr(void*) const {
  AT_ERROR("getDeviceFromPtr not defined for UndefinedType");
}

Storage UndefinedType::storage(bool resizable) const {
  AT_ERROR("storage not defined for UndefinedType");
}
Storage UndefinedType::storage(size_t size, bool resizable) const {
  AT_ERROR("storage(size_t) not defined for UndefinedType");
}
Storage UndefinedType::storageFromBlob(void * data, int64_t size, const std::function<void(void*)> & deleter) const {
  AT_ERROR("storageFromBlob not defined for UndefinedType");
}
Storage UndefinedType::unsafeStorageFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeStorageFromTH not defined for UndefinedType");
}
Storage UndefinedType::storageWithAllocator(int64_t size, Allocator* allocator) const {
  AT_ERROR("storageWithAllocator not defined for UndefinedType");
}
Tensor UndefinedType::unsafeTensorFromTH(void * th_pointer, bool retain) const {
  AT_ERROR("unsafeTensorFromTH not defined for UndefinedType");
}
std::unique_ptr<Generator> UndefinedType::generator() const {
  AT_ERROR("generator not defined for UndefinedType");
}

const char * UndefinedType::toString() const {
  return "UndefinedType";
}

TypeID UndefinedType::ID() const {
  return TypeID::Undefined;
}

size_t UndefinedType::elementSizeInBytes() const {
  AT_ERROR("elementSizeInBytes not defined for UndefinedType");
}

Type & UndefinedType::toBackend(Backend b) const {
  if (b == Backend::Undefined) {
    return TypeDefault::toBackend(b);
  }
  AT_ERROR("toBackend not implemented for UndefinedType to non-UndefinedType");
}
Type & UndefinedType::toScalarType(ScalarType s) const {
  if (s == ScalarType::Undefined) {
    return TypeDefault::toScalarType(s);
  }
  AT_ERROR("toScalarType not implemented for UndefinedType to non-UndefinedType");
}

}
