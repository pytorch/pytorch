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

std::unique_ptr<Generator> ${Type}::generator() const {
  AT_ERROR("generator is not implemented for ${Type}");
}

Backend ${Type}::backend() const {
  return Backend::${Backend};
}

size_t ${Type}::elementSizeInBytes() const {
  AT_ERROR("elementSizeInBytes is not implemented for ${Type}");
}

${type_method_definitions}

} // namespace at
