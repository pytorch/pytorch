#include <ATen/${Type}.h>

namespace at {

${Type}::${Type}() : ${Backend}Type() {}

caffe2::TypeMeta ${Type}::typeMeta() const {
    return caffe2::TypeMeta::Make<${ScalarType}>();
}

const char * ${Type}::toString() const {
  return "${Type}";
}

TypeID ${Type}::ID() const {
  return ${TypeID};
}

} // namespace at
