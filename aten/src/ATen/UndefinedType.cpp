#include <ATen/UndefinedType.h>
#include <c10/util/Exception.h>

namespace at {

UndefinedType::UndefinedType()
    : TypeDefault(UndefinedTensorId(), /*is_variable=*/false, /*is_undefined=*/true) {}
Backend UndefinedType::backend() const {
  return Backend::Undefined;
}

const char * UndefinedType::toString() const {
  return "UndefinedType";
}

TypeID UndefinedType::ID() const {
  return TypeID::Undefined;
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
