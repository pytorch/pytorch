#include <ATen/${Type}.h>

#include <ATen/core/ATenDispatch.h>

namespace at {

std::unordered_map<std::string, void *>& ${Type}Dispatch::get_fn_table() {
  static std::unordered_map<std::string, void *> fn_table;
  return fn_table;
}

${Type}::${Type}()
  : TypeDefault() {}

const char * ${Type}::toString() const {
  return "${Type}";
}

TypeID ${Type}::ID() const {
  return ${TypeID};
}

${type_method_definitions}

static auto& registerer = globalATenDispatch()
  ${function_registrations};

} // namespace at
