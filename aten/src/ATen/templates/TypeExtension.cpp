#include <ATen/${Type}.h>

namespace at {

std::unordered_map<std::string, void *>& ${Type}Dispatch::get_fn_table() {
  static std::unordered_map<std::string, void *> fn_table;
  return fn_table;
}

${Type}::${Type}()
  : TypeDefault(${Backend}TensorId(), /*is_variable=*/false, /*is_undefined=*/false) {}

Allocator* ${Type}::allocator() const {
  AT_ERROR("allocator is not implemented for ${Type}");
}

Device ${Type}::getDeviceFromPtr(void * data) const {
  return DeviceType::${DeviceType};
}

Backend ${Type}::backend() const {
  return Backend::${Backend};
}

const char * ${Type}::toString() const {
  return "${Type}";
}

TypeID ${Type}::ID() const {
  return ${TypeID};
}

${type_method_definitions}

} // namespace at
