#include <ATen/UndefinedType.h>
#include <c10/util/Exception.h>

namespace at {

UndefinedType::UndefinedType()
    : TypeDefault() {}

const char * UndefinedType::toString() const {
  return "UndefinedType";
}

TypeID UndefinedType::ID() const {
  return TypeID::Undefined;
}

}
