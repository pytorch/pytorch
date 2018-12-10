#include <ATen/${Type}.h>

namespace at {

${Type}::${Type}()
  : TypeDefault(${Backend}TensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}

template <typename FnPtr>
std::map<std::string, FnPtr> ${Type}::${Type}Dispatch<FnPtr>::fn_table_;

Allocator* ${Type}::allocator() const {
  AT_ERROR("allocator is not implemented for ${Type}");
}

Device ${Type}::getDeviceFromPtr(void * data) const {
  return DeviceType::${DeviceType};
}

std::unique_ptr<Generator> ${Type}::generator() const {
  AT_ERROR("generator is not implemented for ${Type}");
}

ScalarType ${Type}::scalarType() const {
  AT_ERROR("scalarType is not implemented for ${Type}");
}

caffe2::TypeMeta ${Type}::typeMeta() const {
  AT_ERROR("typeMeta is not implemented for ${Type}");
}

Backend ${Type}::backend() const {
  return Backend::${Backend};
}

const char * ${Type}::toString() const {
  return "${Type}";
}

TypeID ${Type}::ID() const {
  AT_ERROR("ID is not implemented for ${Type}");
}

size_t ${Type}::elementSizeInBytes() const {
  AT_ERROR("elementSizeInBytes is not implemented for ${Type}");
}

${type_derived_method_definitions}

} // namespace at
