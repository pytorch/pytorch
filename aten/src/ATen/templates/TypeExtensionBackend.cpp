#include <ATen/${Type}.h>

namespace at {

template<typename FnPtr>
FnPtr ${Type}Dispatch::get_function(std::string schema) {
  auto fn_table = get_fn_table<FnPtr>();
  auto it = fn_table.find(schema);
  if (it != fn_table.end()) {
    return it->second;
  }
  AT_ERROR("No function implemented for schema: ", schema);
}

template<typename FnPtr>
void ${Type}Dispatch::register_function(std::string schema, FnPtr fn) {
  get_fn_table<FnPtr>()[schema] = fn;
}

template<typename FnPtr>
std::map<std::string, FnPtr>& ${Type}Dispatch::get_fn_table() {
  static std::map<std::string, FnPtr> fn_table;
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
  return ${TypeID};
}

size_t ${Type}::elementSizeInBytes() const {
  AT_ERROR("elementSizeInBytes is not implemented for ${Type}");
}

${type_method_definitions}

} // namespace at
