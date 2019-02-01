#pragma once
#include <ATen/Backend.h>
${extension_backend_headers}

namespace at {

template <typename FnPtr>
inline std::set<std::string>& get_valid_schemas() {
  static std::set<std::string> schemas;
  return schemas;
}

${get_valid_schemas_specializations}

template <typename FnPtr>
inline void register_extension_backend_op(
    Backend backend,
    const char * schema,
    FnPtr fn) {
      auto & valid_schemas = get_valid_schemas<FnPtr>();
      if (valid_schemas.find(schema) == valid_schemas.end()) {
        AT_ERROR("Attempted to register invalid combination of function and schema for: ", schema);
      }
      switch (backend) {
        ${extension_backend_register_switches}
        default:
          AT_ERROR("Invalid extension backend: ", toString(backend));
    }
}

} // namespace at
